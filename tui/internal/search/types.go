package search

type Result struct {
	ID                 int     `json:"id"`
	Title              string  `json:"title"`
	Document           string  `json:"document"`
	Score              float64 `json:"score"`
	Rank               int     `json:"rank"`
	BM25Score          float64 `json:"bm25_score"`
	SemScore           float64 `json:"sem_score"`
	HybridScore        float64 `json:"hybrid_score"`
	RRFScore           float64 `json:"rrf_score"`
	RerankScore        float64 `json:"rerank_score"`
	BatchRank          int     `json:"batch_rank"`
	CrossEncoderScore  float64 `json:"cross_encoder_score"`
	BM25Rank           int     `json:"bm25_rank"`
	SemanticRank       int     `json:"semantic_rank"`
}

type Request struct {
	Mode     string
	Query    string
	Limit    int
	Rerank   string
	Enhance  string
	Alpha    float64
}

type Response struct {
	OK             bool     `json:"ok"`
	Mode           string   `json:"mode"`
	Query          string   `json:"query"`
	EnhancedQuery  string   `json:"enhanced_query,omitempty"`
	Answer         string   `json:"answer,omitempty"`
	Results        []Result `json:"results"`
	Error          string   `json:"error,omitempty"`
}

// Searcher runs a movie search. Implementations may call Python or a mock.
type Searcher interface {
	Search(req Request) (*Response, error)
}
