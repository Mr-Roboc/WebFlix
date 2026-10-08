package ui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"webflix/tui/internal/commands"
	"webflix/tui/internal/search"
)

const defaultLimit = 5
const defaultAlpha = 0.5

var (
	accent      = lipgloss.Color("#79c0ff")
	muted       = lipgloss.Color("#8b949e")
	userColor   = lipgloss.Color("#d2a8ff")
	okColor     = lipgloss.Color("#3fb950")
	errColor    = lipgloss.Color("#f85149")
	borderColor = lipgloss.Color("#58a6ff")

	brandStyle  = lipgloss.NewStyle().Foreground(accent).Bold(true)
	descStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("#e6edf3")).Bold(true)
	hintStyle   = lipgloss.NewStyle().Foreground(muted)
	userStyle   = lipgloss.NewStyle().Foreground(userColor).Bold(true)
	sysStyle    = lipgloss.NewStyle().Foreground(muted)
	errStyle    = lipgloss.NewStyle().Foreground(errColor)
	titleStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("#e6edf3")).Bold(true)
	scoreStyle  = lipgloss.NewStyle().Foreground(okColor)
	selStyle    = lipgloss.NewStyle().Foreground(accent).Bold(true)
	itemStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("#c9d1d9"))
	footerStyle = lipgloss.NewStyle().Foreground(muted)
	boxStyle    = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(borderColor).
			Padding(0, 1)
)

type searchDoneMsg struct {
	resp *search.Response
	err  error
}

type Model struct {
	width      int
	height     int
	input      textinput.Model
	viewport   viewport.Model
	spinner    spinner.Model
	searcher   search.Searcher
	lines      []string
	mode       string
	rerank     string
	enhance    string
	limit      int
	alpha      float64
	searching  bool
	paletteIdx int
	ready      bool
}

func New(searcher search.Searcher) Model {
	ti := textinput.New()
	ti.Placeholder = "Search movies, or type / for commands"
	ti.Prompt = "> "
	ti.PromptStyle = lipgloss.NewStyle().Foreground(accent).Bold(true)
	ti.PlaceholderStyle = hintStyle
	ti.Cursor.Style = lipgloss.NewStyle().Foreground(accent)
	ti.Focus()
	ti.CharLimit = 512

	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = lipgloss.NewStyle().Foreground(accent)

	vp := viewport.New(80, 12)
	vp.SetContent("")

	m := Model{
		input:    ti,
		viewport: vp,
		spinner:  sp,
		searcher: searcher,
		mode:     "hybrid",
		limit:    defaultLimit,
		alpha:    defaultAlpha,
	}
	m.addSystem("Welcome to WebFlix. Type a query to search, or /help for commands.")
	return m
}

func (m Model) Init() tea.Cmd {
	return tea.Batch(textinput.Blink, m.spinner.Tick)
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.ready = true
		m.layout()
		return m, nil

	case spinner.TickMsg:
		if m.searching {
			var cmd tea.Cmd
			m.spinner, cmd = m.spinner.Update(msg)
			cmds = append(cmds, cmd)
		}

	case searchDoneMsg:
		m.searching = false
		m.input.Placeholder = "Search movies, or type / for commands"
		if msg.err != nil {
			m.addError(msg.err.Error())
		} else if msg.resp != nil && !msg.resp.OK {
			m.addError(msg.resp.Error)
		} else if msg.resp != nil {
			m.addResponse(msg.resp)
		}
		m.refreshViewport()
		return m, nil

	case tea.KeyMsg:
		if m.searching && msg.Type != tea.KeyCtrlC {
			return m, nil
		}
		switch msg.Type {
		case tea.KeyCtrlC:
			return m, tea.Quit
		case tea.KeyEsc:
			if m.paletteOpen() {
				m.input.SetValue("")
				m.paletteIdx = 0
				return m, nil
			}
		case tea.KeyCtrlL:
			m.lines = nil
			m.addSystem("Transcript cleared.")
			m.refreshViewport()
			return m, nil
		case tea.KeyUp:
			if m.paletteOpen() {
				if m.paletteIdx > 0 {
					m.paletteIdx--
				}
				return m, nil
			}
			m.viewport.LineUp(1)
			return m, nil
		case tea.KeyDown:
			if m.paletteOpen() {
				filtered := commands.Filter(m.input.Value())
				if m.paletteIdx < len(filtered)-1 {
					m.paletteIdx++
				}
				return m, nil
			}
			m.viewport.LineDown(1)
			return m, nil
		case tea.KeyTab:
			if m.paletteOpen() {
				filtered := commands.Filter(m.input.Value())
				if len(filtered) > 0 {
					cmd := filtered[m.paletteIdx]
					if needsArg(cmd.Name) && !hasArg(m.input.Value(), cmd.Name) {
						m.input.SetValue("/" + cmd.Name + " ")
					} else {
						m.input.SetValue("/" + cmd.Name)
					}
					m.input.CursorEnd()
				}
				return m, nil
			}
		case tea.KeyEnter:
			return m.submit()
		}
	}

	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	cmds = append(cmds, cmd)
	if m.paletteOpen() {
		filtered := commands.Filter(m.input.Value())
		if m.paletteIdx >= len(filtered) {
			m.paletteIdx = 0
		}
	}
	return m, tea.Batch(cmds...)
}

func (m Model) submit() (tea.Model, tea.Cmd) {
	value := strings.TrimSpace(m.input.Value())
	if value == "" {
		return m, nil
	}

	if strings.HasPrefix(value, "/") {
		filtered := commands.Filter(value)
		fields := strings.Fields(strings.TrimPrefix(value, "/"))
		if len(fields) == 1 && len(filtered) > 0 {
			sel := filtered[m.paletteIdx]
			if sel.Name != strings.ToLower(fields[0]) && strings.HasPrefix(sel.Name, strings.ToLower(fields[0])) {
				if needsArg(sel.Name) {
					m.input.SetValue("/" + sel.Name + " ")
					m.input.CursorEnd()
					return m, nil
				}
				value = "/" + sel.Name
			}
		}
		action, err := commands.Parse(value)
		m.input.SetValue("")
		m.paletteIdx = 0
		if err != nil {
			m.addError(err.Error())
			m.refreshViewport()
			return m, nil
		}
		return m.applyAction(action)
	}

	query := value
	m.input.SetValue("")
	m.addUser(query)
	m.searching = true
	m.refreshViewport()
	req := search.Request{
		Mode:    m.mode,
		Query:   query,
		Limit:   m.limit,
		Rerank:  m.rerank,
		Enhance: m.enhance,
		Alpha:   m.alpha,
	}
	return m, tea.Batch(m.spinner.Tick, runSearch(m.searcher, req))
}

func (m Model) applyAction(action commands.Action) (tea.Model, tea.Cmd) {
	if action.Quit {
		return m, tea.Quit
	}
	if action.Clear {
		m.lines = nil
		m.addSystem("Transcript cleared.")
		m.refreshViewport()
		return m, nil
	}
	if action.Help {
		m.addSystem(commands.HelpText())
		m.refreshViewport()
		return m, nil
	}
	if action.SetMode != "" {
		m.mode = action.SetMode
	}
	if action.SetRerank != nil {
		m.rerank = *action.SetRerank
	}
	if action.SetEnhance != nil {
		m.enhance = *action.SetEnhance
	}
	if action.SetLimit != nil {
		m.limit = *action.SetLimit
	}
	if action.SetAlpha != nil {
		m.alpha = *action.SetAlpha
	}
	if action.Message != "" {
		m.addSystem(action.Message)
	}
	m.refreshViewport()
	return m, nil
}

func runSearch(searcher search.Searcher, req search.Request) tea.Cmd {
	return func() tea.Msg {
		resp, err := searcher.Search(req)
		return searchDoneMsg{resp: resp, err: err}
	}
}

func (m Model) View() string {
	if !m.ready {
		return "loading..."
	}
	header := renderHeader(m.width)
	palette := ""
	if m.paletteOpen() {
		palette = m.renderPalette()
	}
	input := boxStyle.Width(max(m.width-2, 20)).Render(m.input.View())
	footer := footerStyle.Render(m.footer())
	parts := []string{header, m.viewport.View()}
	if palette != "" {
		parts = append(parts, palette)
	}
	parts = append(parts, input, footer)
	return lipgloss.JoinVertical(lipgloss.Left, parts...)
}

func renderHeader(width int) string {
	logo := brandStyle.Render(strings.Join([]string{
		"██╗   ██╗███████╗██████╗ ███████╗██╗     ██╗██╗  ██╗",
		"██║   ██║██╔════╝██╔══██╗██╔════╝██║     ██║╚██╗██╔╝",
		"██║ █╗ ██║█████╗  ██████╔╝█████╗  ██║     ██║ ╚███╔╝ ",
		"██║███╗██║██╔══╝  ██╔══██╗██╔══╝  ██║     ██║ ██╔██╗ ",
		"╚███╔███╔╝███████╗██████╔╝██║     ███████╗██║██╔╝ ██╗",
	}, "\n"))
	description := descStyle.Render("movie search engine")
	return strings.Join([]string{
		lipgloss.PlaceHorizontal(width, lipgloss.Center, logo),
		lipgloss.PlaceHorizontal(width, lipgloss.Center, description),
	}, "\n")
}

func (m *Model) layout() {
	chrome := 13
	if m.paletteOpen() {
		chrome += 9
	}
	h := m.height - chrome
	if h < 5 {
		h = 5
	}
	m.viewport.Width = max(m.width-2, 20)
	m.viewport.Height = h
	m.input.Width = max(m.width-8, 10)
	m.refreshViewport()
}

func (m *Model) refreshViewport() {
	m.viewport.SetContent(strings.Join(m.lines, "\n"))
	m.viewport.GotoBottom()
}

func (m Model) paletteOpen() bool {
	v := m.input.Value()
	return strings.HasPrefix(v, "/") && !m.searching
}

func (m Model) renderPalette() string {
	filtered := commands.Filter(m.input.Value())
	if len(filtered) == 0 {
		return hintStyle.Render("  no matching commands")
	}
	if m.paletteIdx >= len(filtered) {
		m.paletteIdx = 0
	}
	var b strings.Builder
	limit := 8
	start := 0
	if m.paletteIdx >= limit {
		start = m.paletteIdx - limit + 1
	}
	end := start + limit
	if end > len(filtered) {
		end = len(filtered)
	}
	for i := start; i < end; i++ {
		cmd := filtered[i]
		line := fmt.Sprintf("/%-12s %s", cmd.Name, cmd.Description)
		if i == m.paletteIdx {
			b.WriteString(selStyle.Render("  ▸ " + line))
		} else {
			b.WriteString(itemStyle.Render("    " + line))
		}
		b.WriteString("\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

func (m Model) footer() string {
	status := fmt.Sprintf(" mode:/%s  limit:%d", m.mode, m.limit)
	if m.mode == "weighted" {
		status += fmt.Sprintf("  alpha:%.2f", m.alpha)
	}
	if m.rerank != "" {
		status += "  rerank:" + m.rerank
	}
	if m.enhance != "" {
		status += "  enhance:" + m.enhance
	}
	if m.searching {
		status = m.spinner.View() + " searching…" + status
	}
	hints := "  / commands  enter search  ctrl+c quit"
	return status + "\n" + hints
}

func (m *Model) addUser(query string) {
	m.lines = append(m.lines, "", userStyle.Render("you › "+query))
}

func (m *Model) addSystem(text string) {
	m.lines = append(m.lines, "", sysStyle.Render(text))
}

func (m *Model) addError(text string) {
	m.lines = append(m.lines, "", errStyle.Render("error: "+text))
}

func (m *Model) addResponse(resp *search.Response) {
	switch resp.Mode {
	case "keyword":
		m.addKeywordResponse(resp)
	case "semantic":
		m.addSemanticResponse(resp)
	case "weighted":
		m.addWeightedResponse(resp)
	case "hybrid":
		m.addHybridResponse(resp)
	case "rag", "summarize", "citation", "question":
		m.addGeneratedResponse(resp)
	default:
		m.addDefaultResponse(resp)
	}
}

func (m *Model) addKeywordResponse(resp *search.Response) {
	m.lines = append(m.lines, sysStyle.Render("Searching for: "+resp.Query))
	for _, r := range resp.Results {
		m.lines = append(m.lines, titleStyle.Render(fmt.Sprintf("%d. (%d) %s", r.Rank, r.ID, r.Title)))
	}
}

func (m *Model) addSemanticResponse(resp *search.Response) {
	for i, r := range resp.Results {
		m.lines = append(m.lines, titleStyle.Render(fmt.Sprintf("%d %s : Score : %v", i, r.Title, r.Score)))
	}
}

func (m *Model) addWeightedResponse(resp *search.Response) {
	for _, r := range resp.Results {
		m.lines = append(m.lines,
			titleStyle.Render(fmt.Sprintf("%d. %s", r.Rank, r.Title)),
			scoreStyle.Render(fmt.Sprintf("  Hybrid Score: %.3f", r.HybridScore)),
			scoreStyle.Render(fmt.Sprintf("  BM25: %.3f, Semantic: %.3f", r.BM25Score, r.SemScore)),
			hintStyle.Render(r.Document),
		)
	}
}

func (m *Model) addHybridResponse(resp *search.Response) {
	m.addEnhancedQuery(resp)
	if m.rerank == "" {
		m.addRRFResults(resp)
		return
	}

	for _, r := range resp.Results {
		lines := []string{fmt.Sprintf("%d. %s", r.Rank, r.Title)}
		switch m.rerank {
		case "individual":
			lines = append(lines,
				fmt.Sprintf("Reranking: %.3f/10", r.RerankScore),
				fmt.Sprintf("RRF Score:%.3f", r.RRFScore),
				fmt.Sprintf("BM25 Rank: %d, Semantic Rank: %d", r.BM25Rank, r.SemanticRank),
			)
		case "batch":
			lines = append(lines,
				fmt.Sprintf("Batch rank: %d", r.BatchRank),
				fmt.Sprintf("RRF Score:%.3f", r.RRFScore),
				fmt.Sprintf("BM25 Rank: %d, Semantic Rank: %d", r.BM25Rank, r.SemanticRank),
			)
		case "cross_encoder":
			lines = append(lines,
				fmt.Sprintf("   Cross Encoder Score: %.3f", r.CrossEncoderScore),
				fmt.Sprintf("   RRF Score: %.3f", r.RRFScore),
				fmt.Sprintf("   BM25 Rank: %d, Semantic Rank: %d", r.BM25Rank, r.SemanticRank),
				"   "+r.Document,
			)
		}
		for _, line := range lines {
			m.lines = append(m.lines, titleStyle.Render(line))
		}
		m.lines = append(m.lines, "")
	}
}

func (m *Model) addGeneratedResponse(resp *search.Response) {
	m.addRRFResults(resp)
	if resp.Mode == "summarize" {
		m.lines = append(m.lines, "", sysStyle.Render(" LLM Summary: "), "")
	}
	if strings.TrimSpace(resp.Answer) != "" {
		m.lines = append(m.lines, wrap(resp.Answer, max(m.width-4, 40)))
	}
}

func (m *Model) addEnhancedQuery(resp *search.Response) {
	if resp.EnhancedQuery == "" || resp.EnhancedQuery == resp.Query {
		return
	}
	arrow := "-->"
	if strings.EqualFold(m.enhance, "spell") {
		arrow = "->"
	}
	m.lines = append(m.lines, sysStyle.Render(fmt.Sprintf(
		"Enhanced query (%s): '%s' %s '%s'",
		strings.ToUpper(m.enhance), resp.Query, arrow, resp.EnhancedQuery,
	)), "")
}

func (m *Model) addRRFResults(resp *search.Response) {
	for _, r := range resp.Results {
		m.lines = append(m.lines,
			titleStyle.Render(fmt.Sprintf("%d. %s", r.Rank, r.Title)),
			scoreStyle.Render(fmt.Sprintf("RRF Score:%.3f", r.RRFScore)),
			sysStyle.Render(fmt.Sprintf("BM25 Rank: %d, Semantic Rank: %d", r.BM25Rank, r.SemanticRank)),
			"",
		)
	}

}

func (m *Model) addDefaultResponse(resp *search.Response) {
	if resp.EnhancedQuery != "" && resp.EnhancedQuery != resp.Query {
		m.lines = append(m.lines, sysStyle.Render("enhanced › "+resp.EnhancedQuery))
	}
	if strings.TrimSpace(resp.Answer) != "" {
		m.lines = append(m.lines, titleStyle.Render("answer"), wrap(resp.Answer, max(m.width-4, 40)))
	}
	for _, r := range resp.Results {
		title := fmt.Sprintf("%d. %s", r.Rank, r.Title)
		if r.ID != 0 {
			title = fmt.Sprintf("%d. (%d) %s", r.Rank, r.ID, r.Title)
		}
		m.lines = append(m.lines, titleStyle.Render(title))
		m.lines = append(m.lines, scoreStyle.Render("   "+scoreLine(r, resp.Mode)))
		if r.Document != "" {
			m.lines = append(m.lines, hintStyle.Render("   "+r.Document))
		}
		m.lines = append(m.lines, "")
	}
}

func scoreLine(r search.Result, mode string) string {
	parts := []string{fmt.Sprintf("score %.3f", r.Score)}
	if r.RRFScore != 0 {
		parts = append(parts, fmt.Sprintf("rrf %.3f", r.RRFScore))
	}
	if r.HybridScore != 0 {
		parts = append(parts, fmt.Sprintf("hybrid %.3f", r.HybridScore))
	}
	if r.BM25Score != 0 {
		parts = append(parts, fmt.Sprintf("bm25 %.3f", r.BM25Score))
	}
	if r.SemScore != 0 {
		parts = append(parts, fmt.Sprintf("sem %.3f", r.SemScore))
	}
	if r.RerankScore != 0 {
		parts = append(parts, fmt.Sprintf("rerank %.1f/10", r.RerankScore))
	}
	if r.CrossEncoderScore != 0 {
		parts = append(parts, fmt.Sprintf("ce %.3f", r.CrossEncoderScore))
	}
	if r.BM25Rank != 0 || r.SemanticRank != 0 {
		parts = append(parts, fmt.Sprintf("ranks bm25:%d sem:%d", r.BM25Rank, r.SemanticRank))
	}
	if r.BatchRank != 0 {
		parts = append(parts, fmt.Sprintf("batch #%d", r.BatchRank))
	}
	_ = mode
	return strings.Join(parts, "  ·  ")
}

func wrap(text string, width int) string {
	text = strings.TrimSpace(text)
	if width < 20 {
		width = 20
	}
	var lines []string
	for _, para := range strings.Split(text, "\n") {
		para = strings.TrimSpace(para)
		if para == "" {
			lines = append(lines, "")
			continue
		}
		for len(para) > width {
			cut := strings.LastIndex(para[:width], " ")
			if cut <= 0 {
				cut = width
			}
			lines = append(lines, para[:cut])
			para = strings.TrimSpace(para[cut:])
		}
		if para != "" {
			lines = append(lines, para)
		}
	}
	return strings.Join(lines, "\n")
}

func needsArg(name string) bool {
	switch name {
	case "rerank", "enhance", "limit", "alpha":
		return true
	default:
		return false
	}
}

func hasArg(value, name string) bool {
	fields := strings.Fields(strings.TrimPrefix(strings.TrimSpace(value), "/"))
	return len(fields) >= 2 && strings.EqualFold(fields[0], name)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
