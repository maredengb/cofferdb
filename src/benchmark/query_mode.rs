// A mode identifies which retrieval channels execute for one benchmark query. Keep
// the wire names and engine-specific score labels stable because saved reports use
// them to identify comparable runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BenchmarkQueryMode {
    // Run only the text retrieval channel and report its BM25-derived final score.
    TextBm25,
    // Run only exact vector retrieval. The benchmark uses the embedding supplied
    // for that query.
    VectorExact,
    // Enable both retrieval channels. The engine combines their contributions into
    // its own final score.
    Hybrid,
}

impl BenchmarkQueryMode {
    // Parse only the stable artifact spellings. The caller determines whether an
    // omitted mode can use a default.
    pub(super) fn parse(value: &str) -> Option<Self> {
        match value {
            "text_bm25" => Some(Self::TextBm25),
            "vector_exact" => Some(Self::VectorExact),
            "hybrid" => Some(Self::Hybrid),
            _ => None,
        }
    }

    // Emit the same token accepted by parse. Reports identify the selected mode by
    // this spelling.
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::TextBm25 => "text_bm25",
            Self::VectorExact => "vector_exact",
            Self::Hybrid => "hybrid",
        }
    }

    // Label the score as computed by CofferDB for the selected mode. The label
    // identifies its origin and meaning without promising a scale shared by
    // another engine.
    pub(super) fn score_kind(self) -> &'static str {
        match self {
            Self::TextBm25 => "cofferdb_bm25_final_score",
            Self::VectorExact => "cofferdb_vector_final_score",
            Self::Hybrid => "cofferdb_hybrid_final_score",
        }
    }

    // Keep SQLite baseline labels separate. They identify the converted FTS5
    // ranking score, normalized cosine similarity, and weighted fusion score
    // reported by the baseline.
    pub(super) fn sqlite_score_kind(self) -> &'static str {
        match self {
            Self::TextBm25 => "sqlite_fts5_bm25",
            Self::VectorExact => "sqlite_vec_cosine",
            Self::Hybrid => "sqlite_fts5_vec_weighted_sum",
        }
    }

    // Decide whether this run executes text retrieval. A query row can carry text
    // even when a particular mode does not execute that channel.
    pub(super) fn has_text(self) -> bool {
        matches!(self, Self::TextBm25 | Self::Hybrid)
    }

    // Decide whether execution needs vector input. The same predicate applies to a
    // vector-only run and the vector side of a hybrid run.
    pub(super) fn has_vector(self) -> bool {
        matches!(self, Self::VectorExact | Self::Hybrid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_query_modes_keep_wire_names_branches_and_engine_local_score_labels() {
        for (name, mode, text, vector, coffer_score, sqlite_score) in [
            (
                "text_bm25",
                BenchmarkQueryMode::TextBm25,
                true,
                false,
                "cofferdb_bm25_final_score",
                "sqlite_fts5_bm25",
            ),
            (
                "vector_exact",
                BenchmarkQueryMode::VectorExact,
                false,
                true,
                "cofferdb_vector_final_score",
                "sqlite_vec_cosine",
            ),
            (
                "hybrid",
                BenchmarkQueryMode::Hybrid,
                true,
                true,
                "cofferdb_hybrid_final_score",
                "sqlite_fts5_vec_weighted_sum",
            ),
        ] {
            assert_eq!(BenchmarkQueryMode::parse(name), Some(mode));
            assert_eq!(mode.as_str(), name);
            assert_eq!(mode.has_text(), text);
            assert_eq!(mode.has_vector(), vector);
            assert_eq!(mode.score_kind(), coffer_score);
            assert_eq!(mode.sqlite_score_kind(), sqlite_score);
            for seed in 0..100 {
                for invalid in [
                    format!("{name}_{seed}"),
                    format!("{seed}_{name}"),
                    format!("{name}{}", " ".repeat(seed + 1)),
                    format!("{}{name}", " ".repeat(seed + 1)),
                    format!("{name}\0{seed}"),
                ] {
                    assert_eq!(BenchmarkQueryMode::parse(&invalid), None);
                }
            }
        }
        for invalid in ["", "TEXT_BM25", "VectorExact", "bm25", "vector", "Hybrid"] {
            assert_eq!(BenchmarkQueryMode::parse(invalid), None);
        }
    }
}
