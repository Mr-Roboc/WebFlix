package commands

import "testing"

func TestParseModes(t *testing.T) {
	action, err := Parse("/semantic")
	if err != nil {
		t.Fatal(err)
	}
	if action.SetMode != "semantic" {
		t.Fatalf("got mode %q", action.SetMode)
	}
}

func TestParseRerank(t *testing.T) {
	action, err := Parse("/rerank cross_encoder")
	if err != nil {
		t.Fatal(err)
	}
	if action.SetRerank == nil || *action.SetRerank != "cross_encoder" {
		t.Fatalf("unexpected rerank: %#v", action.SetRerank)
	}
}

func TestParseUnknown(t *testing.T) {
	if _, err := Parse("/nope"); err == nil {
		t.Fatal("expected error")
	}
}
