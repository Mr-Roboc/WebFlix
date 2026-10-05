package search

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const searchTimeout = 5 * time.Minute

type PythonSearcher struct {
	Root string
}

func FindRepoRoot() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	dir := cwd
	for {
		if _, err := os.Stat(filepath.Join(dir, "cli", "tui_search.py")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return "", fmt.Errorf("could not find WebFlix repo root from %s (missing cli/tui_search.py)", cwd)
}

func (p *PythonSearcher) Search(req Request) (*Response, error) {
	ctx, cancel := context.WithTimeout(context.Background(), searchTimeout)
	defer cancel()

	args := []string{
		filepath.Join("cli", "tui_search.py"),
		"--json",
		"--mode", req.Mode,
		"--query", req.Query,
		"--limit", strconv.Itoa(req.Limit),
		"--alpha", strconv.FormatFloat(req.Alpha, 'f', -1, 64),
	}
	if req.Rerank != "" {
		args = append(args, "--rerank", req.Rerank)
	}
	if req.Enhance != "" {
		args = append(args, "--enhance", req.Enhance)
	}

	cmd, err := pythonCommand(ctx, args)
	if err != nil {
		return nil, err
	}
	cmd.Dir = p.Root

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	runErr := cmd.Run()
	raw := bytes.TrimSpace(stdout.Bytes())
	if len(raw) == 0 {
		msg := strings.TrimSpace(stderr.String())
		if runErr != nil {
			if errors.Is(runErr, context.DeadlineExceeded) {
				return nil, fmt.Errorf("search timed out")
			}
			if msg == "" {
				msg = runErr.Error()
			}
		}
		if msg == "" {
			msg = "search adapter returned no JSON"
		}
		return &Response{OK: false, Error: msg}, nil
	}

	var resp Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("invalid JSON from search adapter: %w\n%s", err, raw)
	}
	if !resp.OK && resp.Error == "" && runErr != nil {
		resp.Error = runErr.Error()
	}
	return &resp, nil
}

func pythonCommand(ctx context.Context, scriptArgs []string) (*exec.Cmd, error) {
	if _, err := exec.LookPath("uv"); err == nil {
		args := append([]string{"run", "python"}, scriptArgs...)
		return exec.CommandContext(ctx, "uv", args...), nil
	}
	python, err := exec.LookPath("python3")
	if err != nil {
		python, err = exec.LookPath("python")
		if err != nil {
			return nil, fmt.Errorf("neither uv nor python3 is on PATH")
		}
	}
	return exec.CommandContext(ctx, python, scriptArgs...), nil
}
