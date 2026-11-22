/// Resource presets with fixed default candidate windows and cache budgets.
/// String inputs accept only `expanded`, `balanced`, and `constrained`.
/// These names do not promise retrieval quality, latency, or process RSS.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ResourceProfile {
    /// 500 candidates per stage and a 512 MiB default cache budget.
    Expanded,
    /// Default preset with 200 candidates per stage and a 256 MiB default cache budget.
    #[default]
    Balanced,
    /// 50 candidates per stage and a 64 MiB default cache budget.
    Constrained,
}

impl ResourceProfile {
    /// Return the lowercase profile name used in JSON results and configuration.
    pub fn as_json_str(self) -> &'static str {
        // Keep the stable wire names used in persisted settings and benchmark
        // artifacts in step with parsing. The numeric preset values can be
        // recovered from the enum after the spelling has been checked.
        match self {
            ResourceProfile::Expanded => "expanded",
            ResourceProfile::Balanced => "balanced",
            ResourceProfile::Constrained => "constrained",
        }
    }

    /// Return the lowercase profile name accepted by the CLI.
    pub fn as_cli_str(self) -> &'static str {
        match self {
            ResourceProfile::Expanded => "expanded",
            ResourceProfile::Balanced => "balanced",
            ResourceProfile::Constrained => "constrained",
        }
    }

    /// Parse exactly `expanded`, `balanced` or `constrained`. Unknown names return `None`.
    pub fn parse(value: &str) -> Option<Self> {
        // Leave fallback policy to the caller. An unrecognized value returns None
        // rather than selecting Balanced, which would make a mistyped explicit
        // setting look like an omitted setting.
        match value {
            "expanded" => Some(ResourceProfile::Expanded),
            "balanced" => Some(ResourceProfile::Balanced),
            "constrained" => Some(ResourceProfile::Constrained),
            _ => None,
        }
    }

    /// Return this preset's candidate limits, cache budget and import batch size.
    pub fn params(self) -> ProfileParams {
        // Keep search windows, cache allowance and import batch target together
        // for each preset. They govern different resources: a per-query top_k may
        // raise a candidate window, while import scheduling uses its own batch
        // setting.
        match self {
            ResourceProfile::Expanded => ProfileParams {
                bm25_candidates: 500,
                vector_candidates: 500,
                fusion_candidates: 500,
                cache_budget_mb: 512,
                background_compaction: false,
                import_batch_chunks: 2_000,
            },
            ResourceProfile::Balanced => ProfileParams {
                bm25_candidates: 200,
                vector_candidates: 200,
                fusion_candidates: 200,
                cache_budget_mb: 256,
                background_compaction: false,
                import_batch_chunks: 1_000,
            },
            ResourceProfile::Constrained => ProfileParams {
                bm25_candidates: 50,
                vector_candidates: 50,
                fusion_candidates: 50,
                cache_budget_mb: 64,
                background_compaction: false,
                import_batch_chunks: 250,
            },
        }
    }
}

/// Default candidate windows, memory budget and import batch size for a resource profile.
/// The three candidate counts are preset defaults. At search time each effective
/// window is at least the requested `top_k`; the profile alone is insufficient to
/// reconstruct a particular query's limits.
#[derive(Clone, Copy, Debug)]
pub struct ProfileParams {
    /// Maximum number of candidates retained by lexical retrieval.
    pub bm25_candidates: usize,
    /// Maximum number of candidates retained by vector retrieval.
    pub vector_candidates: usize,
    /// Maximum number of candidates retained by fusion.
    pub fusion_candidates: usize,
    /// Cache budget in MiB.
    pub cache_budget_mb: usize,
    /// Whether to request background compaction. Currently only false is supported.
    pub background_compaction: bool,
    /// Import batch target in records. The planner also applies a byte target and
    /// keeps dependent records in atomic groups. A batch can include document,
    /// chunk and embedding records.
    pub import_batch_chunks: usize,
}

/// Synchronization policy for committed writes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Durability {
    /// Request file synchronization at commit.
    #[default]
    Commit,
    /// Skip per-commit file synchronization. Recent writes may be lost on a system crash.
    Fast,
}

impl Durability {
    /// Return the policy name used in configuration.
    pub fn as_str(self) -> &'static str {
        match self {
            Durability::Commit => "commit",
            Durability::Fast => "fast",
        }
    }

    /// Parse `commit` or `fast`, returning `None` for any other spelling.
    pub fn parse(value: &str) -> Option<Self> {
        // Keep the accepted spelling exact. An unknown durability setting must
        // reach the caller as a parse failure so a typo cannot silently select a
        // different write synchronization policy.
        match value {
            "commit" => Some(Durability::Commit),
            "fast" => Some(Durability::Fast),
            _ => None,
        }
    }
}

/// I/O preference for opening or creating a database. The current backend uses file reads
/// for all three settings.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum IoMode {
    /// Use the default I/O path.
    #[default]
    Auto,
    /// Request read-based access.
    Read,
    /// Request access with memory mapping disabled.
    MmapDisabled,
}

/// Creation settings. Defaults to the balanced profile, commit durability and automatic I/O.
#[derive(Clone, Debug)]
pub struct DatabaseConfig {
    /// Resource preset used unless overridden for an operation.
    pub resource_profile: ResourceProfile,
    /// Cache budget override in MiB, or `None` to use the profile default.
    pub cache_budget_mb: Option<usize>,
    /// Whether to request background compaction. Currently only false is supported.
    pub background_compaction: bool,
    /// Synchronization policy for writes.
    pub durability: Durability,
    /// Requested I/O mode.
    pub io_mode: IoMode,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        // Leave the cache override unset. Code that later selects a resource
        // profile can then use that profile's cache allowance; storing today's
        // numeric default here would turn it into an explicit override.
        Self {
            resource_profile: ResourceProfile::Balanced,
            cache_budget_mb: None,
            background_compaction: false,
            // The durability default applies at commit boundaries. I/O mode is an
            // independent opening preference, so changing that option alone must
            // not opt a writer into fast durability.
            durability: Durability::Commit,
            io_mode: IoMode::Auto,
        }
    }
}

impl DatabaseConfig {
    // Validate the JSON form of typed runtime settings using the same
    // business-value limits as other configuration input. The database separately
    // decides whether requested features, such as background compaction, are
    // supported.
    pub(crate) fn validate_json_policy(&self) -> crate::Result<()> {
        use crate::json::{object_from_pairs, JsonInputKind, JsonValue};
        object_from_pairs([
            (
                "resource_profile",
                JsonValue::String(self.resource_profile.as_json_str().into()),
            ),
            (
                "cache_budget_mb",
                // Null preserves the distinction between inheriting the profile's
                // allowance and requesting zero MiB. Validation does not resolve
                // the effective cache budget or rewrite the typed value.
                self.cache_budget_mb
                    .map(|value| JsonValue::Number(value as f64))
                    .unwrap_or(JsonValue::Null),
            ),
            (
                "background_compaction",
                JsonValue::Bool(self.background_compaction),
            ),
            (
                "durability",
                JsonValue::String(self.durability.as_str().into()),
            ),
            (
                "io_mode",
                // Represent the requested I/O preference with its configuration
                // spelling. All three values currently use file reads; the
                // projected setting records the caller's selection, not a
                // different backend.
                JsonValue::String(
                    match self.io_mode {
                        IoMode::Auto => "auto",
                        IoMode::Read => "read",
                        IoMode::MmapDisabled => "mmap_disabled",
                    }
                    .into(),
                ),
            ),
        ])
        // Apply runtime-config limits to the complete object after all fields are
        // present. This check performs no file access and does not apply the
        // settings to an open database.
        .validate_kind(JsonInputKind::RuntimeConfig)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_presets_keep_their_parameter_contract_and_balanced_default() {
        assert_eq!(ResourceProfile::default(), ResourceProfile::Balanced);
        assert_eq!(
            DatabaseConfig::default().resource_profile,
            ResourceProfile::Balanced
        );
        for (profile, name, candidates, cache, batch) in [
            (ResourceProfile::Expanded, "expanded", 500, 512, 2_000),
            (ResourceProfile::Balanced, "balanced", 200, 256, 1_000),
            (ResourceProfile::Constrained, "constrained", 50, 64, 250),
        ] {
            assert_eq!(ResourceProfile::parse(name), Some(profile));
            assert_eq!(profile.as_cli_str(), name);
            assert_eq!(profile.as_json_str(), name);
            let params = profile.params();
            assert_eq!(params.bm25_candidates, candidates);
            assert_eq!(params.vector_candidates, candidates);
            assert_eq!(params.fusion_candidates, candidates);
            assert_eq!(params.cache_budget_mb, cache);
            assert_eq!(params.import_batch_chunks, batch);
            assert!(!params.background_compaction);
        }
    }

    #[test]
    fn parse_rejects_old_names_case_variants_whitespace_and_help() {
        for value in [
            "quality-first",
            "quality_first",
            "resource-first",
            "resource_first",
            "performance",
            "eco",
            "QualityFirst",
            "ResourceFirst",
            "Expanded",
            "Balanced",
            "Constrained",
            "EXPANDED",
            "BALANCED",
            "CONSTRAINED",
            " expanded",
            "expanded ",
            " balanced",
            "constrained ",
            "constrained\n",
            "expanded-budget",
            "help",
            "",
            "turbo",
        ] {
            assert_eq!(ResourceProfile::parse(value), None, "{value:?}");
        }
    }
}
