package main

import (
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"

	"webflix/tui/internal/search"
	"webflix/tui/internal/ui"
)

func main() {
	root, err := search.FindRepoRoot()
	if err != nil {
		fmt.Fprintf(os.Stderr, "webflix: %v\n", err)
		os.Exit(1)
	}
	searcher := &search.PythonSearcher{Root: root}
	p := tea.NewProgram(ui.New(searcher), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "webflix: %v\n", err)
		os.Exit(1)
	}
}
