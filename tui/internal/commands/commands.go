package commands

import (
	"fmt"
	"strconv"
	"strings"
)

type Kind int

const (
	KindMode Kind = iota
	KindModifier
	KindUtility
)

type Command struct {
	Name        string
	Usage       string
	Description string
	Kind        Kind
}

type Action struct {
	SetMode    string
	SetRerank  *string
	SetEnhance *string
	SetLimit   *int
	SetAlpha   *float64
	Help       bool
	Clear      bool
	Quit       bool
	Message    string
}

var Catalog = []Command{
	{Name: "keyword", Usage: "/keyword", Description: "BM25 keyword search", Kind: KindMode},
	{Name: "semantic", Usage: "/semantic", Description: "Embedding semantic search", Kind: KindMode},
	{Name: "hybrid", Usage: "/hybrid", Description: "RRF hybrid search (default)", Kind: KindMode},
	{Name: "weighted", Usage: "/weighted", Description: "Alpha-weighted hybrid search", Kind: KindMode},
	{Name: "rag", Usage: "/rag", Description: "Retrieve and generate an answer", Kind: KindMode},
	{Name: "summarize", Usage: "/summarize", Description: "Summarize retrieved movies", Kind: KindMode},
	{Name: "citation", Usage: "/citation", Description: "Answer with citations", Kind: KindMode},
	{Name: "question", Usage: "/question", Description: "Conversational Q&A over results", Kind: KindMode},
	{Name: "rerank", Usage: "/rerank individual|batch|cross_encoder|none", Description: "Rerank hybrid/RAG results", Kind: KindModifier},
	{Name: "enhance", Usage: "/enhance spell|rewrite|expand|none", Description: "Rewrite the query before search", Kind: KindModifier},
	{Name: "limit", Usage: "/limit N", Description: "Number of results to show", Kind: KindModifier},
	{Name: "alpha", Usage: "/alpha 0.5", Description: "BM25 vs semantic weight (weighted mode)", Kind: KindModifier},
	{Name: "help", Usage: "/help", Description: "Show available commands", Kind: KindUtility},
	{Name: "clear", Usage: "/clear", Description: "Clear the transcript", Kind: KindUtility},
	{Name: "quit", Usage: "/quit", Description: "Exit WebFlix", Kind: KindUtility},
}

func Filter(query string) []Command {
	q := strings.TrimSpace(query)
	q = strings.TrimPrefix(q, "/")
	fields := strings.Fields(q)
	prefix := ""
	if len(fields) > 0 {
		prefix = strings.ToLower(fields[0])
	}
	if prefix == "" {
		return append([]Command(nil), Catalog...)
	}
	var out []Command
	for _, cmd := range Catalog {
		if strings.HasPrefix(cmd.Name, prefix) || cmd.Name == prefix {
			out = append(out, cmd)
		}
	}
	return out
}

func HelpText() string {
	var b strings.Builder
	b.WriteString("Commands\n")
	for _, cmd := range Catalog {
		b.WriteString(fmt.Sprintf("  %-48s %s\n", cmd.Usage, cmd.Description))
	}
	return strings.TrimRight(b.String(), "\n")
}

func Parse(line string) (Action, error) {
	line = strings.TrimSpace(line)
	if !strings.HasPrefix(line, "/") {
		return Action{}, fmt.Errorf("not a command")
	}
	fields := strings.Fields(line[1:])
	if len(fields) == 0 {
		return Action{Help: true}, nil
	}
	name := strings.ToLower(fields[0])
	args := fields[1:]

	switch name {
	case "keyword", "semantic", "hybrid", "weighted", "rag", "summarize", "citation", "question":
		return Action{SetMode: name, Message: "Mode set to /" + name}, nil
	case "rerank":
		if len(args) == 0 {
			return Action{}, fmt.Errorf("usage: /rerank individual|batch|cross_encoder|none")
		}
		v := strings.ToLower(args[0])
		switch v {
		case "none", "off":
			empty := ""
			return Action{SetRerank: &empty, Message: "Rerank disabled"}, nil
		case "individual", "batch", "cross_encoder":
			return Action{SetRerank: &v, Message: "Rerank set to " + v}, nil
		default:
			return Action{}, fmt.Errorf("usage: /rerank individual|batch|cross_encoder|none")
		}
	case "enhance":
		if len(args) == 0 {
			return Action{}, fmt.Errorf("usage: /enhance spell|rewrite|expand|none")
		}
		v := strings.ToLower(args[0])
		switch v {
		case "none", "off":
			empty := ""
			return Action{SetEnhance: &empty, Message: "Query enhance disabled"}, nil
		case "spell", "rewrite", "expand":
			return Action{SetEnhance: &v, Message: "Enhance set to " + v}, nil
		default:
			return Action{}, fmt.Errorf("usage: /enhance spell|rewrite|expand|none")
		}
	case "limit":
		if len(args) == 0 {
			return Action{}, fmt.Errorf("usage: /limit N")
		}
		n, err := strconv.Atoi(args[0])
		if err != nil || n <= 0 {
			return Action{}, fmt.Errorf("limit must be a positive integer")
		}
		return Action{SetLimit: &n, Message: fmt.Sprintf("Limit set to %d", n)}, nil
	case "alpha":
		if len(args) == 0 {
			return Action{}, fmt.Errorf("usage: /alpha 0.0-1.0")
		}
		a, err := strconv.ParseFloat(args[0], 64)
		if err != nil || a < 0 || a > 1 {
			return Action{}, fmt.Errorf("alpha must be a number between 0 and 1")
		}
		return Action{SetAlpha: &a, Message: fmt.Sprintf("Alpha set to %.2f", a)}, nil
	case "help":
		return Action{Help: true}, nil
	case "clear":
		return Action{Clear: true}, nil
	case "quit", "exit", "q":
		return Action{Quit: true}, nil
	default:
		return Action{}, fmt.Errorf("unknown command: /%s", name)
	}
}
