//! Search request construction and result reporting.
//!
//! The collection search in `crate::db` resolves a read view, retrieves candidates
//! and fuses their scores. This module owns the public request and report types,
//! resolves effective weights, and turns evidence captured during that search into
//! JSON.
//!
//! Rust callers can retain owned rows after the search reservation is released.
//! Binding and CLI callers can carry the reservation into JSON construction so the
//! peak includes both the remaining rows and the output being assembled.
use crate::error::{Error, ErrorCode, Result};
#[cfg(test)]
use crate::index::text::Bm25Hit;
use crate::index::text::MatchedTerm;
#[cfg(test)]
use crate::json::object_from_pairs;
use crate::json::{JsonBuilder, JsonInputKind, JsonReservation, JsonValue};
use crate::metadata::MetadataBoostConfig;
use crate::model::{Chunk, CollectionHandle, QueryEmbeddingInput, SearchWeights};
use crate::profile::ResourceProfile;

// The report encodes top_k as a JSON number backed by f64. Past 2^53 - 1 two
// adjacent positive integers can have the same representation. On a narrower host
// every usize already fits in that range.
pub(crate) const MAX_SEARCH_TOP_K: usize = if usize::BITS > 53 {
    9_007_199_254_740_991u64 as usize
} else {
    usize::MAX
};

// Bindings validate the same bound as the search itself. Check it before candidate
// windows are widened to top_k, so an unrepresentable request can never become a
// limit in a result or diagnosis.
pub(crate) fn validate_search_top_k(top_k: usize) -> Result<()> {
    if top_k == 0 || top_k > MAX_SEARCH_TOP_K {
        return Err(Error::new(
            ErrorCode::InvalidSearchParams,
            "top_k must be a positive integer in the supported exact JSON range",
        )
        .with_detail("field", "top_k")
        .with_detail("min", 1)
        .with_detail("max", MAX_SEARCH_TOP_K)
        .with_detail("actual", top_k));
    }
    Ok(())
}

// A builder holds one request. Execution resolves the request against the
// collection, retrieves bounded text and vector windows, scores their union and
// materializes the highest-ranked chunks. Each returned row keeps both its scores
// and the evidence used to produce them.
/// Builds a text, vector or hybrid search on one collection. Call [`execute`](Self::execute) to run it.
///
/// See the [Rust programming guide](https://docs.rs/crate/cofferdb/0.2.0/source/docs/rust.md) for a write, search and reopen example.
#[derive(Clone)]
pub struct SearchBuilder {
    // Keeps the database handle and the collection ID together until execution
    // resolves the collection.
    collection: CollectionHandle,
    // Own the caller's text until execution can trim it. A blank string leaves only
    // the vector branch available.
    text: Option<String>,
    // Both vector setters replace this slot. The structured form also carries the
    // dimension and optional model identity checked against the collection.
    vector: Option<QueryEmbeddingInput>,
    // Keep the predicate as JSON until execution can validate it with the query's
    // resource budget. The default empty object admits every candidate.
    filter: JsonValue,
    // None defers the choice of weights until the available retrieval branches and
    // boost rules are known.
    weights: Option<SearchWeights>,
    // The caller can suspend boosting while retaining the configuration and its
    // identity in diagnostics.
    metadata_boost_enabled: bool,
    // The validated rule set supplies both the scoring rules and the digest
    // reported for this request.
    metadata_boost_config: Option<MetadataBoostConfig>,
    // Maximum returned hits. Each retrieval and fusion window is at least this
    // large, even when the profile's default window is smaller.
    top_k: usize,
    // An explicit override applies to this search. None reads the database's
    // profile when the search runs.
    resource_profile: Option<ResourceProfile>,
}

/// Ranked hits and the context used to produce them.
#[derive(Clone, Debug)]
pub struct SearchResults {
    /// Version of the JSON report format.
    /// Identifies the report schema independently of the crate version.
    pub schema_version: String,
    /// Name of the collection described by this value.
    pub collection: String,
    /// Query text, when a text query was supplied.
    /// Search trims leading and trailing whitespace. A blank input leaves this
    /// field as None even when a vector produces hits.
    pub query: Option<String>,
    /// Requested maximum number of returned chunks.
    /// The fused candidate window can contain more entries than this limit, and a
    /// search can return fewer hits.
    pub top_k: usize,
    /// Resource profile used for the search.
    /// A per-query override takes precedence over the database setting.
    pub profile: ResourceProfile,
    /// Settings and execution evidence for this query.
    /// Limits, observed work and index provenance describe the whole search and are
    /// shared by its hits.
    pub query_context: SearchQueryContext,
    /// Returned chunks in rank order, best first.
    /// Each hit owns its identifiers, metadata and explanation and can outlive the
    /// search builder.
    pub results: Vec<SearchResult>,
}

/// Observed work for one search, including vector comparisons and retained fusion candidates.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SearchExecutionStats {
    /// Number of vector entries scanned, including supplemental work during fusion.
    /// An entry visited and rejected by the filter can count as a scan without
    /// contributing any distance work.
    pub vector_scan_count: usize,
    /// Number of vector components scored in recall and score completion. Scoring
    /// one 128-dimensional embedding adds 128, regardless of which metric the
    /// collection uses.
    pub distance_ops: usize,
    /// Number of candidates retained after fusion, before limiting results to top-k.
    /// Each retained chunk counts once, including a chunk recalled by both text and
    /// vector.
    pub fusion_candidate_count: usize,
}

/// One ranked chunk, its component scores and the evidence for its ranking.
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// One-based position in the returned results.
    /// Equal final scores are ordered by the internal chunk ID.
    pub rank: usize,
    /// Database-assigned identifier of the chunk.
    pub chunk_id: String,
    /// Caller-assigned chunk identifier scoped to this collection.
    pub chunk_external_id: String,
    /// Database-assigned identifier of the document.
    pub document_id: String,
    /// Caller-assigned external ID of the owning document.
    pub document_external_id: String,
    /// Preview of the matching chunk text.
    /// The preview starts at the beginning of the chunk and includes at most 240
    /// Unicode scalar values.
    pub snippet: String,
    /// Application metadata from the owning document.
    /// All hits from a document carry its metadata as an owned JSON value.
    pub document_metadata: JsonValue,
    /// Application metadata from the matching chunk.
    /// This is the chunk's own object, kept separately so a key can also exist on
    /// the document.
    pub chunk_metadata: JsonValue,
    /// Combined score used to rank this hit.
    /// An unnormalized dot product can take the weighted sum outside the unit
    /// interval.
    pub final_score: f64,
    /// Raw lexical score when available for this hit.
    /// With a text query, a chunk recalled only by vector can have a lexical score
    /// of zero. Without a text query the score is absent.
    pub bm25_score: Option<f64>,
    /// Raw metric-specific similarity or distance when available for this hit.
    /// A hit recalled by text may acquire this score during score completion. A hit
    /// without a usable vector score retains an absent value.
    pub vector_score: Option<f64>,
    /// Metadata rule contribution before applying its fusion weight.
    /// Matching rule boosts are added and capped by the configuration before
    /// fusion.
    pub metadata_boost: f64,
    /// Component values before applying fusion weights.
    pub normalized_scores: NormalizedScores,
    /// Structured evidence for this hit, including retrieval, scoring and filtering.
    /// The evidence includes the bounded windows that recalled this hit, the
    /// components used in its final score and the predicate evaluated against it.
    pub explain: JsonValue,
}

/// Component values used for weighted fusion. Their scales depend on the retrieval metric.
#[derive(Clone, Copy, Debug, Default)]
pub struct NormalizedScores {
    /// BM25 score scaled relative to the query's normalization candidates.
    /// The denominator is the maximum over the union of the two recall windows,
    /// before the fusion limit. With no positive maximum the component is zero.
    pub bm25_norm: f64,
    /// Metric-specific vector score for fusion, when available.
    /// An unnormalized dot product keeps its raw value. An absent score contributes
    /// zero to fusion and remains absent in the report.
    pub vector_norm: Option<f64>,
    /// Metadata rule contribution before applying its fusion weight.
    /// This is the capped sum of matching rules for the hit, including zero when no
    /// boost applies.
    pub metadata_boost: f64,
}

/// Requested and effective fusion weights, including any adjustment made for this query.
#[derive(Clone, Debug)]
pub struct WeightContext {
    /// Weights requested by the caller or chosen by default.
    /// Preserves the original mix even when a requested branch is unavailable at
    /// execution.
    pub requested: SearchWeights,
    /// Weights actually applied to component scores.
    /// Unavailable branches have zero weight and any remaining positive weights
    /// retain their relative proportions.
    pub effective: SearchWeights,
    /// Identifies how the requested weights were selected.
    /// Distinguishes an explicit override from the defaults selected with or
    /// without active boost rules.
    pub source: String,
    /// Reason for adjusting weights, if an adjustment was recorded.
    /// Absent when the requested weights pass through without an adjustment.
    pub renormalization_reason: Option<String>,
}

/// Effective query settings, observed candidate counts and index provenance.
#[derive(Clone, Debug)]
pub struct SearchQueryContext {
    /// Whether a text query was supplied.
    /// An input consisting only of whitespace is treated as absent.
    pub text_present: bool,
    /// Whether a query embedding was supplied.
    /// Presence describes the request even when the collection contributes no
    /// scored embeddings.
    pub embedding_present: bool,
    /// Whether a metadata filter applies to this query.
    /// An empty object or JSON null leaves the filter absent.
    pub filter_present: bool,
    /// Origin, requested values and effective values of the fusion weights.
    /// A zero component in an effective mix is insufficient to tell whether its
    /// branch was absent or was given zero weight explicitly.
    pub weights: WeightContext,
    /// Whether a metadata boost configuration was supplied.
    /// An explicitly disabled configuration still has a rule count and digest.
    pub metadata_boost_present: bool,
    /// Whether boosting is enabled with a configuration containing at least one rule.
    /// This describes eligibility for the search. Individual hits can still match
    /// no rules.
    pub metadata_boost_enabled: bool,
    /// Canonical configuration digest, when a boost configuration is present.
    /// The SHA-256 digest covers the canonical boost JSON alone. Requests with
    /// different text or vectors can share this value.
    pub boost_config_hash: Option<String>,
    /// Number of configured boost rules, including unmatched rules.
    /// A disabled rule set retains its configured count.
    pub boost_rule_count: usize,
    /// Requested maximum number of returned chunks.
    pub top_k: usize,
    /// Resource profile used for the search.
    /// The profile supplies candidate-window defaults independently of the reported
    /// work counters.
    pub profile: ResourceProfile,
    /// Configured candidate limits for each retrieval stage.
    /// Each limit is the larger of the selected profile's window and the requested
    /// top-k.
    pub candidate_limits: CandidateLimits,
    /// Actual candidate counts retained at each retrieval stage.
    /// Per-branch counts can overlap. Fusion counts distinct retained chunks before
    /// the final top-k cutoff.
    pub candidate_counts: CandidateLimits,
    /// Observed vector work and number of retained fusion candidates.
    pub execution_stats: SearchExecutionStats,
    /// Identifies the persistent checkpoint or the transient index used.
    /// Describes the index path finally used, including a fallback after a
    /// checkpoint failed validation.
    pub index_source: String,
    /// Reason for rebuilding the search index in memory, when applicable.
    /// A usable checkpoint leaves this absent. A fallback to committed records can
    /// set it during score completion.
    pub index_rebuild_reason: Option<String>,
    /// Details of any in-memory rebuild used by this search.
    /// Optional counters distinguish a rebuild that measured zero work from a
    /// checkpoint query that skipped the rebuild.
    pub transient_rebuild: TransientRebuildContext,
}

/// Input for constructing a query context. The constructors initialize execution counts to zero.
#[derive(Clone, Debug)]
pub struct SearchQueryContextInput {
    /// Whether a text query was supplied.
    /// Set after trimming the query text so whitespace alone does not select BM25
    /// defaults.
    pub text_present: bool,
    /// Whether a query embedding was supplied.
    pub embedding_present: bool,
    /// Whether a metadata filter applies to this query.
    /// Set from the parsed predicate. An empty object and JSON null both represent
    /// an unrestricted search.
    pub filter_present: bool,
    /// Origin, requested values and effective values of the fusion weights.
    pub weights: WeightContext,
    /// Whether a metadata boost configuration was supplied.
    pub metadata_boost_present: bool,
    /// Whether boosting is enabled with a configuration containing at least one rule.
    /// Set from the enable switch together with a supplied, nonempty rule set.
    pub metadata_boost_enabled: bool,
    /// Canonical configuration digest, when a boost configuration is present.
    /// Identifies the validated boost JSON and is retained even when its rules are
    /// disabled.
    pub boost_config_hash: Option<String>,
    /// Number of configured boost rules, including unmatched rules.
    pub boost_rule_count: usize,
    /// Requested maximum number of returned chunks.
    pub top_k: usize,
    /// Resource profile used for the search.
    pub profile: ResourceProfile,
    /// Configured candidate limits for each retrieval stage.
    /// Supply the windows after taking the maximum of each profile default and the
    /// requested top-k.
    pub candidate_limits: CandidateLimits,
    /// Cache budget in MiB.
    /// The database's explicit cache limit takes precedence over the selected
    /// profile's default.
    pub cache_budget_mb: usize,
    /// Number of chunks scanned to rebuild the index.
    /// Used only for a transient context. Execution can refine this initial count
    /// once it knows which scan ran.
    pub chunks_scanned: usize,
    /// Number of bytes read during rebuilding, when recorded.
    /// The checkpoint constructor ignores this measurement. For a transient
    /// context, zero is a recorded byte count and absence means no measurement was
    /// supplied.
    pub bytes_read: Option<u64>,
}

/// Evidence about an in-memory rebuild from committed records for this query.
#[derive(Clone, Debug)]
pub struct TransientRebuildContext {
    /// Whether an in-memory rebuild started.
    /// Checkpoint queries leave this false unless the search falls back to
    /// committed records.
    pub started: bool,
    /// Whether the in-memory rebuild finished.
    pub completed: bool,
    /// Whether the in-memory rebuild was cancelled.
    /// A successful transient context describes a completed scan. An error can
    /// prevent delivery of a report at all.
    pub cancelled: bool,
    /// Cache budget recorded for the rebuild in MiB.
    pub budget_mb: usize,
    /// Number of chunks scanned by the rebuild, when applicable.
    /// Absent for a checkpoint. A vector-only transient search reports the number
    /// of vector entries inspected during recall.
    pub chunks_scanned: Option<usize>,
    /// Number of bytes read during rebuilding, when recorded.
    /// Counts cumulative payload bytes in the transient search, which can exceed
    /// the memory needed at any one time.
    pub bytes_read: Option<u64>,
    /// Whether the rebuild reports exceeding its budget.
    /// A memory refusal during a search is returned as an error before a successful
    /// search report is delivered.
    pub budget_exceeded: bool,
}

/// Inputs from one materialized hit and the query that ranked it. Borrowed data
/// only has to live through construction of the explanation. The filter already
/// owns its evaluated JSON trace.
#[derive(Clone, Debug)]
pub struct ExplainInput<'a> {
    /// Collection name shown to the caller alongside the database-assigned IDs.
    pub collection_name: &'a str,
    /// Live chunk supplying its ID, its owning document ID and its original source
    /// coordinates.
    pub chunk: &'a Chunk,
    /// Application's ID for the owning document, resolved from the same collection
    /// as the hit.
    pub document_external_id: &'a str,
    /// Owning document's metadata. A string in its `path` field takes precedence
    /// when formatting the source location.
    pub document_metadata: &'a JsonValue,
    /// Optional URI from the document record, also used as the display path when
    /// metadata supplies no string path.
    pub document_source_uri: Option<&'a str>,
    /// True when the surviving bounded BM25 window contained this chunk, regardless
    /// of its eventual lexical weight.
    pub bm25_recalled: bool,
    /// True when the bounded vector window recalled this chunk. A supplemental
    /// vector lookup leaves it false.
    pub vector_recalled: bool,
    /// Number of distinct chunk IDs in the union of the bounded recall windows
    /// before fusion truncation.
    pub normalization_candidate_count: usize,
    /// Completed lexical score. With an active text query a vector-only hit with no
    /// lexical match has a measured score of zero.
    pub bm25_raw: Option<f64>,
    /// Lexical contribution before weighting. Zero when the candidate union has no
    /// positive BM25 maximum.
    pub bm25_norm: f64,
    /// Maximum lexical score in the candidate union. Absent for a vector-only query
    /// and zero when text was supplied but the union has no positive score.
    pub bm25_max: Option<f64>,
    /// Metric result for this hit, including a value obtained during score
    /// completion. Absent if no vector score is available.
    pub vector_raw: Option<f64>,
    /// Vector component actually supplied to fusion. An absent score stays absent
    /// in the report.
    pub vector_norm: Option<f64>,
    /// Sum of this hit's matching rule boosts after applying the configuration's
    /// maximum and before applying its weight.
    pub metadata_boost: f64,
    /// Score already used to rank the hit. The explanation reports the same value
    /// as the result row.
    pub final_score: f64,
    /// Term statistics for this hit from the lexical window or its supplemental
    /// lookup. An empty slice produces empty term arrays.
    pub matched_terms: &'a [MatchedTerm],
    /// Query-wide requested and effective coefficients, including why they differ
    /// when a branch was unavailable.
    pub weights: &'a WeightContext,
    /// Profile selected for this search, including any override on the builder.
    pub resource_profile: ResourceProfile,
    /// Effective stage windows. These are upper bounds even when the corresponding
    /// route supplied no hits.
    pub candidate_limits: CandidateLimits,
    /// Whole-query count of vector entries inspected, including supplementary
    /// lookups for text hits.
    pub vector_scan_count: usize,
    /// Total vector components scored by recall and score completion, shared by all
    /// hit explanations.
    pub distance_ops: usize,
    /// Number of distinct candidates retained by fusion before the final top-k
    /// cutoff.
    pub fusion_candidate_count: usize,
    /// Provenance selected by the search after checkpoint validation and any
    /// fallback.
    pub index_source: &'a str,
    /// Explanation for reading from committed records on fallback. Absent when the
    /// checkpoint remained usable.
    pub index_rebuild_reason: Option<&'a str>,
    /// Query-wide scan measurements copied into the hit explanation, including
    /// optional payload byte counts.
    pub transient_rebuild: TransientRebuildContext,
    /// Collection metric that gives meaning to a raw vector score, such as
    /// Euclidean distance.
    pub distance_metric: &'a str,
    /// Collection's declared normalization setting. It selects the score transform
    /// for dot product.
    pub vectors_normalized: bool,
    /// Owned predicate and evaluation evidence for this chunk, moved into the
    /// output object.
    pub filter: JsonValue,
    /// Whether the caller supplied a boost rule set, including a set that was
    /// subsequently disabled.
    pub metadata_boost_present: bool,
    /// True if the supplied rule set was enabled and contained rules available for
    /// scoring.
    pub metadata_boost_enabled: bool,
    /// Canonical digest borrowed from the query context, or absent if no rule set
    /// was supplied.
    pub boost_config_hash: Option<&'a str>,
    /// Size of the query's configured rule set, independent of which rules matched
    /// this hit.
    pub boost_rule_count: usize,
    /// Number of rules that matched this chunk. Reaching the boost cap does not
    /// reduce the count.
    pub matched_boost_rule_count: usize,
}

/// Counts for the BM25, vector and fusion stages. The enclosing field distinguishes limits from
/// observed counts.
#[derive(Clone, Copy, Debug)]
pub struct CandidateLimits {
    /// Number of lexical candidates at the BM25 stage.
    /// For an observed count, includes only hits surviving the metadata filter on
    /// the bounded lexical window.
    pub bm25: usize,
    /// Number of embedding candidates at the vector stage.
    /// For an observed count, counts the result of vector recall. A supplemental
    /// vector score for a lexical hit adds no vector candidate.
    pub vector: usize,
    /// Number of candidates at the fusion stage.
    /// For an observed count, includes distinct retained chunks from either window
    /// and can exceed the number of returned hits.
    pub fusion: usize,
}

/// Diagnostic report for one target chunk under a particular search request.
#[derive(Clone, Debug)]
pub struct WhyNotDiagnosis {
    /// Version of the JSON report format.
    pub schema_version: String,
    /// Scope of the diagnosis, currently `debug_exact`.
    pub diagnosis_scope: String,
    /// Indicates exact diagnostic mode. The target rank scan is skipped when the
    /// target is missing, deleted, belongs to another collection or fails the
    /// filter.
    pub debug_full_scan: bool,
    /// Resource preset used for this operation.
    pub resource_profile: ResourceProfile,
    /// Text part of the diagnosed query, if supplied.
    /// Comes from the ordinary search, so a blank or unsupplied text query is
    /// absent.
    pub query_text: Option<String>,
    /// Settings and execution evidence for this query.
    /// Describes the ordinary bounded search used to determine the returned rank.
    /// Target-specific evidence is obtained separately.
    /// An intervening write can occur between the bounded search and the target
    /// observation. Their evidence can therefore describe different committed
    /// views.
    pub query_context: SearchQueryContext,
    /// Structured outcome for the target, including its reason and supporting evidence.
    /// Includes target identity, eligibility, ranks and limits. A target that never
    /// reached ranking can still have a diagnosis.
    pub result: JsonValue,
}

impl SearchBuilder {
    /// Start with no inputs, no boost or filter, default weights, the database's profile and a
    /// top-k of ten.
    // Start with a request that has no scoring inputs. Builders can fill it
    // incrementally; execution validates the final combination under the
    // collection's current view.
    pub fn new(collection: CollectionHandle) -> Self {
        Self {
            collection,
            text: None,
            vector: None,
            filter: JsonValue::object(),
            weights: None,
            metadata_boost_enabled: false,
            metadata_boost_config: None,
            top_k: 10,
            resource_profile: None,
        }
    }

    /// Set the text used for BM25 retrieval. Add a vector to run a hybrid search.
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Set a query vector without a model name. Its length must match the collection's dimension.
    /// Setting this after a structured embedding replaces that embedding and its
    /// optional model name.
    pub fn vector(mut self, vector: Vec<f32>) -> Self {
        self.vector = Some(QueryEmbeddingInput::from_vector(vector));
        self
    }

    /// Set a query vector with explicit dimension and optional model identity.
    pub fn query_embedding(mut self, embedding: QueryEmbeddingInput) -> Self {
        self.vector = Some(embedding);
        self
    }

    /// Set the JSON metadata filter. Its expression is validated when the search runs.
    pub fn filter(mut self, filter: JsonValue) -> Self {
        self.filter = filter;
        self
    }

    /// Set the requested BM25, vector and metadata weights used during fusion.
    pub fn weights(mut self, weights: SearchWeights) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Enable or disable the configured metadata boost for this search.
    pub fn metadata_boost_enabled(mut self, enabled: bool) -> Self {
        // Keep the switch independent of the configuration. Disabling an installed
        // set still lets reports identify the supplied rules and lets the caller
        // re-enable them.
        self.metadata_boost_enabled = enabled;
        self
    }

    /// Install a validated metadata rule set and enable it for this search.
    pub fn metadata_boost(mut self, config: MetadataBoostConfig) -> Self {
        // Installing a configuration also enables it, so the order of this method
        // and an explicit disable determines whether the stored rules participate.
        self.metadata_boost_enabled = true;
        self.metadata_boost_config = Some(config);
        self
    }

    /// Set the maximum number of hits, default ten. Execution requires a positive value in the
    /// supported exact JSON integer range.
    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Override the database's resource preset for this query.
    pub fn resource_profile(mut self, profile: ResourceProfile) -> Self {
        self.resource_profile = Some(profile);
        self
    }

    /// Validate the request and return ranked, owned hits and query diagnostics. Supply text, an
    /// embedding or both.
    ///
    /// # Errors
    ///
    /// Returns an error if the request is invalid, the database is closed, or
    /// the search cannot read its inputs or fit its working set in the budget.
    ///
    /// # Examples
    ///
    /// See the [crate-level example](crate) for a search over committed chunks.
    pub fn execute(self) -> Result<SearchResults> {
        self.execute_with_json_context(None)
    }

    // The ordinary Rust API returns owned rows. A caller that will immediately
    // convert them to budgeted JSON can instead keep the reservation returned by
    // execute_retained while both representations are live.
    pub(crate) fn execute_with_json_context(
        self,
        context: Option<crate::json::JsonOperationContext>,
    ) -> Result<SearchResults> {
        // Returning just the Rust value drops the operation's reservation here. Its
        // hit strings and JSON trees are already owned by the result.
        self.execute_retained(context).map(|(results, _)| results)
    }

    pub(crate) fn execute_retained(
        self,
        context: Option<crate::json::JsonOperationContext>,
    ) -> Result<(SearchResults, crate::storage::RecordReservation)> {
        // Resolve the inputs together at execution. Only then are the collection's
        // vector constraints, the database's default profile and the set of
        // available scoring branches known.
        self.collection.db.search_in_collection(
            &self.collection.collection_id,
            self.text.as_deref(),
            self.vector.as_ref(),
            &self.filter,
            self.weights,
            self.metadata_boost_enabled,
            self.metadata_boost_config.as_ref(),
            self.top_k,
            self.resource_profile,
            context,
        )
    }
}

impl SearchResults {
    /// Return the JSON representation of this search results.
    pub fn to_json(&self) -> JsonValue {
        // The borrowed public conversion leaves the caller's rows usable. The
        // consuming path below can instead move the result's existing allocations
        // into the JSON object.
        self.clone().into_json()
    }

    pub(crate) fn into_json(self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    pub(crate) fn into_json_retained(
        mut self,
        work: &mut crate::storage::RecordReservation,
    ) -> Result<JsonValue> {
        // While rows become JSON, two sets of allocations are live: the
        // report-level strings still in self, and the contents of the row currently
        // being converted. Save the size of the surviving wrapper before taking the
        // rows.
        let wrapper_bytes = self.wrapper_workspace_bytes();
        let rows = std::mem::take(&mut self.results);
        let results = {
            // Transfer owned row strings from workspace credit to JSON credit as
            // each string moves. Metadata and explanation trees already have JSON
            // credit in this reservation. Reserving them again would make
            // conversion fail near the budget even though their contents were never
            // copied.
            let (json, strings) = work.parts_mut();
            let builder =
                JsonBuilder::budgeted_owned_strings(json, strings, JsonInputKind::StoredJson);
            // If a row fails admission, drop the partially built array and
            // remaining rows. The caller still owns the reservation, and this
            // early return skips the workspace trims below.
            builder.array(rows.into_iter().map(|result| result.build_json(&builder)))?
        };
        // The old result array is now freed; its strings moved into JSON without copying.
        work.retain_workspace(wrapper_bytes);
        // The array now owns the converted rows. Build the outer object last with
        // the remaining wrapper strings, so taking ownership of a field can
        // transfer its existing capacity.
        let value = {
            let (json, strings) = work.parts_mut();
            self.build_json_fields(
                results,
                &JsonBuilder::budgeted_owned_strings(json, strings, JsonInputKind::StoredJson),
            )
        };
        // The wrapper was consumed even if a later JSON key could not be admitted.
        // Release its remaining workspace credit on either outcome.
        work.retain_workspace(0);
        value
    }

    pub(crate) fn wrapper_workspace_bytes(&self) -> usize {
        // Rows retain their own allocations. Count only the string capacities still
        // owned by the wrapper once its rows leave. Their lengths or serialized
        // sizes need not cover those allocations.
        self.schema_version.capacity()
            + self.collection.capacity()
            + self.query.as_ref().map_or(0, String::capacity)
            + self.query_context.workspace_bytes()
    }

    pub(crate) fn discard_rows(&mut self, work: &mut crate::storage::RecordReservation) {
        // Diagnosis can keep the ordinary search context after recording its ranks.
        // Drop any remaining rows before giving back their JSON and workspace
        // credit. Benchmark reporting also takes this path after moving out the
        // external IDs.
        drop(std::mem::take(&mut self.results));
        work.json.retain(0);
        work.retain_workspace(self.wrapper_workspace_bytes());
    }

    // Keep both JSON entry points on the same report envelope. This host-owned
    // path needs no workspace credit to be transferred while rows are consumed.
    fn build_json(mut self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        let results = builder.array(
            std::mem::take(&mut self.results)
                .into_iter()
                .map(|result| result.build_json(builder)),
        )?;
        self.build_json_fields(results, builder)
    }

    fn build_json_fields(self, results: JsonValue, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("schema_version", builder.owned_string(self.schema_version)?),
            ("collection", builder.owned_string(self.collection)?),
            ("query", builder.optional_string(self.query)?),
            ("top_k", JsonValue::Number(self.top_k as f64)),
            ("profile", builder.string(self.profile.as_json_str())?),
            // The top-level counts describe the bounded branch windows and retained
            // fusion rows. The stats include scans and scored components outside
            // those returned windows. Serialize both from the finished query
            // context.
            (
                "candidate_counts",
                candidate_limits_json_in(self.query_context.candidate_counts, builder)?,
            ),
            (
                "execution_stats",
                self.query_context.execution_stats.build_json(builder)?,
            ),
            ("query_context", self.query_context.build_json(builder)?),
            ("results", results),
        ])
    }
}

impl SearchResult {
    /// Return the JSON representation of this search result.
    pub fn to_json(&self) -> JsonValue {
        self.clone()
            .build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    fn build_json(self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        // The IDs and snippet move out as strings. Both metadata objects and the
        // explanation are already JSON trees, so putting them into the row
        // preserves their nested allocations and the credit held for them.
        builder.object([
            ("rank", JsonValue::Number(self.rank as f64)),
            ("chunk_id", builder.owned_string(self.chunk_id)?),
            (
                "chunk_external_id",
                builder.owned_string(self.chunk_external_id)?,
            ),
            ("document_id", builder.owned_string(self.document_id)?),
            (
                "document_external_id",
                builder.owned_string(self.document_external_id)?,
            ),
            ("snippet", builder.owned_string(self.snippet)?),
            ("document_metadata", self.document_metadata),
            ("chunk_metadata", self.chunk_metadata),
            ("final_score", JsonValue::Number(self.final_score)),
            // A null raw score records the absence of that component. A measured
            // zero retains the number zero even though the two cases can make the
            // same contribution to the final score.
            (
                "bm25_score",
                self.bm25_score
                    .map(JsonValue::Number)
                    .unwrap_or(JsonValue::Null),
            ),
            (
                "vector_score",
                self.vector_score
                    .map(JsonValue::Number)
                    .unwrap_or(JsonValue::Null),
            ),
            ("metadata_boost", JsonValue::Number(self.metadata_boost)),
            // The public row names the components bm25 and vector. The explanation
            // names the corresponding values bm25_norm and vector_norm to
            // distinguish them from the raw scores.
            (
                "normalized_scores",
                builder.object([
                    ("bm25", JsonValue::Number(self.normalized_scores.bm25_norm)),
                    (
                        "vector",
                        self.normalized_scores
                            .vector_norm
                            .map(JsonValue::Number)
                            .unwrap_or(JsonValue::Null),
                    ),
                    (
                        "metadata_boost",
                        JsonValue::Number(self.normalized_scores.metadata_boost),
                    ),
                ])?,
            ),
            ("explain", self.explain),
        ])
    }
}

impl SearchQueryContext {
    /// Construct diagnostic context describing a completed in-memory rebuild from committed
    /// data. This constructor performs no I/O.
    pub fn transient_from_committed_snapshot(input: SearchQueryContextInput) -> Self {
        Self::from_checkpoint_with(input, false, |value| Ok(value.to_owned()))
            .expect("host DTO construction is infallible")
    }

    /// Construct diagnostic context describing a query using a persisted checkpoint. This
    /// constructor performs no I/O.
    pub fn persistent_checkpoint(input: SearchQueryContextInput) -> Self {
        Self::from_checkpoint_with(input, true, |value| Ok(value.to_owned()))
            .expect("host DTO construction is infallible")
    }

    pub(crate) fn from_checkpoint_with(
        input: SearchQueryContextInput,
        persistent: bool,
        mut copy: impl FnMut(&str) -> Result<String>,
    ) -> Result<Self> {
        // Create the report after choosing a usable index path. The same
        // constructor serves callers building a host-owned context and searches
        // that must reserve every diagnostic string under an operation budget.
        Ok(Self {
            text_present: input.text_present,
            embedding_present: input.embedding_present,
            filter_present: input.filter_present,
            weights: input.weights,
            metadata_boost_present: input.metadata_boost_present,
            metadata_boost_enabled: input.metadata_boost_enabled,
            boost_config_hash: input.boost_config_hash,
            boost_rule_count: input.boost_rule_count,
            top_k: input.top_k,
            profile: input.profile,
            candidate_limits: input.candidate_limits,
            // Limits are known before recall. Start observed counters at zero and
            // let the search fill them after score completion and fusion, before it
            // materializes the top-k rows.
            candidate_counts: CandidateLimits {
                bm25: 0,
                vector: 0,
                fusion: 0,
            },
            execution_stats: SearchExecutionStats::default(),
            // Use the actual source chosen by the search. These labels can be
            // requested from either a host allocator or the operation's workspace
            // reservation, so creating them can itself fail on the budgeted path.
            index_source: copy(if persistent {
                "persistent_checkpoint"
            } else {
                "transient_rebuild"
            })?,
            index_rebuild_reason: (!persistent)
                .then(|| copy("in_memory_index_from_committed_snapshot"))
                .transpose()?,
            // A checkpoint query still reports its configured budget, but has no
            // transient scan or byte measurement. A completed transient scan can
            // legitimately report zero, so keep absence distinct from that value.
            transient_rebuild: TransientRebuildContext {
                started: !persistent,
                completed: !persistent,
                cancelled: false,
                budget_mb: input.cache_budget_mb,
                chunks_scanned: (!persistent).then_some(input.chunks_scanned),
                bytes_read: if persistent { None } else { input.bytes_read },
                budget_exceeded: false,
            },
        })
    }

    pub(crate) fn workspace_bytes(&self) -> usize {
        // These string allocations move into the report at different stages. Keep
        // their workspace capacities together so discarding ranked rows can release
        // everything except the surviving context.
        self.weights.source.capacity()
            + self
                .weights
                .renormalization_reason
                .as_ref()
                .map_or(0, String::capacity)
            + self.boost_config_hash.as_ref().map_or(0, String::capacity)
            + self.index_source.capacity()
            + self
                .index_rebuild_reason
                .as_ref()
                .map_or(0, String::capacity)
    }

    /// Return the JSON representation of this search query context.
    pub fn to_json(&self) -> JsonValue {
        self.clone()
            .build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    pub(crate) fn build_json(self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("text_present", JsonValue::Bool(self.text_present)),
            // Only the text route uses the tokenizer. A vector-only context reports
            // null here even though individual hit explanations describe the
            // tokenizer configuration for their term fields.
            (
                "tokenizer",
                if self.text_present {
                    crate::TokenizerConfig::default().build_json(builder)?
                } else {
                    JsonValue::Null
                },
            ),
            ("embedding_present", JsonValue::Bool(self.embedding_present)),
            ("filter_present", JsonValue::Bool(self.filter_present)),
            // Preserve how the weights were chosen. An explicit override can be
            // numerically identical to a default, so the source label determines
            // the present flag.
            (
                "weights",
                builder.object([
                    (
                        "present",
                        JsonValue::Bool(self.weights.source == "user_override"),
                    ),
                    ("source", builder.string(&self.weights.source)?),
                    (
                        "requested",
                        weights_json_in(self.weights.requested, builder)?,
                    ),
                    (
                        "effective",
                        weights_json_in(self.weights.effective, builder)?,
                    ),
                    (
                        "renormalization_reason",
                        self.weights
                            .renormalization_reason
                            .as_deref()
                            .map(|reason| builder.string(reason))
                            .transpose()?
                            .unwrap_or_default(),
                    ),
                ])?,
            ),
            // The report exposes the weights both in the nested weights object and
            // as sibling fields of the query context. Build the nested copy first,
            // then move the owned source and reason into the sibling fields.
            (
                "requested_weights",
                weights_json_in(self.weights.requested, builder)?,
            ),
            (
                "effective_weights",
                weights_json_in(self.weights.effective, builder)?,
            ),
            ("weights_source", builder.owned_string(self.weights.source)?),
            (
                "renormalization_reason",
                builder.optional_string(self.weights.renormalization_reason)?,
            ),
            // Presence, enablement and identity have different lifetimes. A
            // disabled rule set still has a digest and a rule count. Only a
            // nonempty enabled set contributes an available metadata weight.
            (
                "metadata_boost_present",
                JsonValue::Bool(self.metadata_boost_present),
            ),
            (
                "metadata_boost_enabled",
                JsonValue::Bool(self.metadata_boost_enabled),
            ),
            (
                "boost_config_hash",
                builder.optional_string(self.boost_config_hash)?,
            ),
            (
                "boost_rule_count",
                JsonValue::Number(self.boost_rule_count as f64),
            ),
            ("top_k", JsonValue::Number(self.top_k as f64)),
            ("profile", builder.string(self.profile.as_json_str())?),
            // This object carries the planned stage limits. The search envelope
            // emits the observed branch counts separately, while execution_stats
            // describes the actual scan and fusion work.
            (
                "candidate_limits",
                candidate_limits_json_in(self.candidate_limits, builder)?,
            ),
            ("execution_stats", self.execution_stats.build_json(builder)?),
            ("index_source", builder.owned_string(self.index_source)?),
            (
                "index_rebuild_reason",
                builder.optional_string(self.index_rebuild_reason)?,
            ),
            (
                "transient_rebuild",
                self.transient_rebuild.build_json(builder)?,
            ),
        ])
    }
}

impl WhyNotDiagnosis {
    /// Return the JSON representation of this why not diagnosis.
    pub fn to_json(&self) -> JsonValue {
        Self {
            schema_version: self.schema_version.clone(),
            diagnosis_scope: self.diagnosis_scope.clone(),
            debug_full_scan: self.debug_full_scan,
            resource_profile: self.resource_profile,
            query_text: self.query_text.clone(),
            query_context: self.query_context.clone(),
            // Only members of an object can become fields of the diagnosis. A
            // manually constructed non-object contributes no fields, so the
            // borrowed projection can omit the cost of cloning it.
            result: self
                .result
                .as_object()
                .map(|_| self.result.clone())
                .unwrap_or_default(),
        }
        .into_json()
    }

    pub(crate) fn into_json(self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    #[cfg(test)]
    pub(crate) fn into_json_in(self, work: &mut JsonReservation) -> Result<JsonValue> {
        self.result
            .reconcile_output_retained_in(crate::json::JsonOutputSchema::WhyNot, work)?;
        self.build_json(&JsonBuilder::budgeted(work, JsonInputKind::StoredJson))
    }

    pub(crate) fn into_json_retained(
        self,
        work: &mut crate::storage::RecordReservation,
    ) -> Result<JsonValue> {
        // The target-specific evidence is already a JSON tree. Reconcile its actual
        // retained size with the operation reservation before adding envelope
        // fields, since earlier stages may have held additional JSON credit.
        self.result
            .reconcile_output_retained_in(crate::json::JsonOutputSchema::WhyNot, &mut work.json)?;
        let value = {
            let (json, strings) = work.parts_mut();
            self.build_json(&JsonBuilder::budgeted_owned_strings(
                json,
                strings,
                JsonInputKind::StoredJson,
            ))
        };
        // All remaining Rust strings belonged to the consumed report. Give back
        // workspace left by fields that were dropped or whose values the existing
        // JSON object supplied.
        work.retain_workspace(0);
        value
    }

    fn build_json(mut self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        // Take the existing object so the nested filter trace and target evidence
        // retain their allocations. Starting from an empty object also gives a
        // valid envelope for a manually constructed non-object result.
        let mut object = self
            .result
            .as_object_mut()
            .map(std::mem::take)
            .unwrap_or_default();
        drop(self.result);
        // Target evidence already present in the object wins, including values for
        // envelope keys. Lazily building absent members matters under a budget:
        // rebuilding a discarded context would consume credit and could fail a
        // valid projection.
        builder.insert_if_absent(&mut object, "schema_version", || {
            builder.owned_string(self.schema_version)
        })?;
        builder.insert_if_absent(&mut object, "diagnosis_scope", || {
            builder.owned_string(self.diagnosis_scope)
        })?;
        builder.insert_if_absent(&mut object, "debug_full_scan", || {
            Ok(JsonValue::Bool(self.debug_full_scan))
        })?;
        builder.insert_if_absent(&mut object, "resource_profile", || {
            builder.string(self.resource_profile.as_json_str())
        })?;
        // Fill the two compatibility keys independently. Existing evidence for one
        // spelling does not supply the value of the other.
        builder.insert_if_absent(&mut object, "profile", || {
            builder.string(self.resource_profile.as_json_str())
        })?;
        // The query text is added to the generated context for this report. If the
        // evidence already supplies a query_context, preserve that entire object,
        // including any existing text.
        builder.insert_if_absent(&mut object, "query_context", || {
            let mut value = self.query_context.build_json(builder)?;
            builder.insert_if_absent(value.as_object_mut().unwrap(), "text", || {
                builder.optional_string(self.query_text)
            })?;
            Ok(value)
        })?;
        Ok(JsonValue::Object(object))
    }
}

impl SearchExecutionStats {
    /// Return the JSON representation of this search execution stats.
    pub fn to_json(self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    // Serialize measured work independently of window sizes. Vector recall may
    // score entries that never reach the retained window, and score completion can
    // perform comparisons for entries recalled through text. The counters stay as
    // usize during search and are binary64 numbers in this projection.
    fn build_json(self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            (
                "vector_scan_count",
                JsonValue::Number(self.vector_scan_count as f64),
            ),
            ("distance_ops", JsonValue::Number(self.distance_ops as f64)),
            (
                "fusion_candidate_count",
                JsonValue::Number(self.fusion_candidate_count as f64),
            ),
        ])
    }
}

impl TransientRebuildContext {
    /// Return the JSON representation of this transient rebuild context.
    pub fn to_json(&self) -> JsonValue {
        self.build_json(&JsonBuilder::host())
            .expect("host DTO construction is infallible")
    }

    fn build_json(&self, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
        builder.object([
            ("started", JsonValue::Bool(self.started)),
            ("completed", JsonValue::Bool(self.completed)),
            ("cancelled", JsonValue::Bool(self.cancelled)),
            ("budget_mb", JsonValue::Number(self.budget_mb as f64)),
            // Null distinguishes a checkpoint with no rebuild measurements from a
            // transient scan over an empty collection. Keep an explicit zero if the
            // rebuild recorded one.
            (
                "chunks_scanned",
                self.chunks_scanned
                    .map(|count| JsonValue::Number(count as f64))
                    .unwrap_or(JsonValue::Null),
            ),
            (
                "bytes_read",
                self.bytes_read
                    .map(|bytes| JsonValue::Number(bytes as f64))
                    .unwrap_or(JsonValue::Null),
            ),
            ("budget_exceeded", JsonValue::Bool(self.budget_exceeded)),
        ])
    }
}

#[cfg(test)]
fn effective_weights(
    requested: Option<SearchWeights>,
    has_text: bool,
    has_vector: bool,
    metadata_boost_available: bool,
) -> Result<WeightContext> {
    effective_weights_with(
        requested,
        has_text,
        has_vector,
        metadata_boost_available,
        |value| Ok(value.to_owned()),
    )
}

pub(crate) fn effective_weights_with(
    requested: Option<SearchWeights>,
    has_text: bool,
    has_vector: bool,
    metadata_boost_available: bool,
    mut copy: impl FnMut(&str) -> Result<String>,
) -> Result<WeightContext> {
    // Choose the source label before applying defaults: an explicit override equal
    // to the default numbers is still user-supplied for reporting purposes.
    let source = if requested.is_some() {
        "user_override"
    } else if metadata_boost_available {
        "default_with_boost"
    } else {
        "default_no_boost"
    };
    let requested = requested
        .unwrap_or_else(|| default_weights_for(has_text, has_vector, metadata_boost_available));
    // Validate the supplied mix before removing unavailable branches. A negative or
    // non-finite value is invalid even when its branch will be disabled.
    requested.validate()?;
    // Diagnostics retain the requested mix. Remove unavailable branches before
    // normalizing so the scores actually combined receive all the weight.
    let mut effective = requested;
    // Availability belongs to the query inputs and configuration. A text route that
    // finds no terms for one hit still keeps its query-wide coefficient. A zero
    // weight on a missing route requires no removal reason.
    let removed_text = !has_text && effective.bm25 > 0.0;
    let removed_vector = !has_vector && effective.vector > 0.0;
    let removed_metadata = !metadata_boost_available && effective.metadata > 0.0;
    if removed_text {
        effective.bm25 = 0.0;
    }
    if removed_vector {
        effective.vector = 0.0;
    }
    if removed_metadata {
        effective.metadata = 0.0;
    }
    // A valid request can lose all its weight here. For example, assigning
    // everything to metadata with no active rules leaves no way to score a text
    // hit. Return the original request and branch availability in the error before
    // reserving any diagnostic strings.
    let sum = effective.bm25 + effective.vector + effective.metadata;
    if sum <= 0.0 {
        return Err(Error::new(
            ErrorCode::InvalidSearchWeights,
            "effective search weights sum to zero",
        )
        .with_detail("field", "weights")
        .with_detail("reason", "effective_sum_zero")
        .with_detail("weights_reason", "effective_sum_zero")
        .with_detail("has_text", has_text)
        .with_detail("has_vector", has_vector)
        .with_detail("metadata_boost_available", metadata_boost_available)
        .with_detail("requested_bm25", requested.bm25)
        .with_detail("requested_vector", requested.vector)
        .with_detail("requested_metadata", requested.metadata));
    }
    // The input validator allows a small rounding tolerance. Preserve an already
    // unit-sum mix within machine epsilon. Otherwise scale each remaining weight by
    // the same factor. A text-only request that assigned 0.2 to text and 0.8 to
    // vector gives text a final weight of 1.0.
    let normalized = (sum - 1.0).abs() > f64::EPSILON;
    if normalized {
        effective.bm25 /= sum;
        effective.vector /= sum;
        effective.metadata /= sum;
    }
    // Report removals in text, vector, metadata order so the reason is stable when
    // two branches disappear. Selecting a static reason lets a budgeted caller
    // reserve only the final strings, without a temporary joined message.
    let reason = match (removed_text, removed_vector, removed_metadata) {
        (true, true, true) => unreachable!("zero effective sum rejected above"),
        (true, true, false) => Some("text branch unavailable; vector branch unavailable"),
        (true, false, true) => Some("text branch unavailable; metadata boost unavailable"),
        (false, true, true) => Some("vector branch unavailable; metadata boost unavailable"),
        (true, false, false) => Some("text branch unavailable"),
        (false, true, false) => Some("vector branch unavailable"),
        (false, false, true) => Some("metadata boost unavailable"),
        (false, false, false) => normalized.then_some("weights normalized to sum 1.0"),
    };
    // Delay the fallible copies until all numeric checks pass. The caller can then
    // account for exactly one source label and, only when present, one adjustment
    // reason in the result.
    Ok(WeightContext {
        requested,
        effective,
        source: copy(source)?,
        renormalization_reason: reason.map(&mut copy).transpose()?,
    })
}

pub(crate) fn default_weights_for(
    has_text: bool,
    has_vector: bool,
    metadata_boost_available: bool,
) -> SearchWeights {
    // An active boost changes the default mix even if no rule matches a particular
    // hit. Choose the weights once for the query so the hit's match outcome changes
    // its score without changing how other candidates are ranked.
    if !metadata_boost_available {
        return SearchWeights::default_for(has_text, has_vector);
    }
    // With one retrieval input the remaining mass goes to that route. With both,
    // the preset reserves a share for the boost and gives the vector route a
    // slightly larger share. A rule can affect the order only of chunks already in
    // the recall union.
    match (has_text, has_vector) {
        (true, true) => SearchWeights {
            bm25: 0.45,
            vector: 0.50,
            metadata: 0.05,
        },
        (true, false) => SearchWeights {
            bm25: 0.95,
            vector: 0.0,
            metadata: 0.05,
        },
        (false, true) => SearchWeights {
            bm25: 0.0,
            vector: 0.95,
            metadata: 0.05,
        },
        (false, false) => SearchWeights::default_for(false, false),
    }
}

#[cfg(test)]
fn build_explain(input: ExplainInput<'_>) -> JsonValue {
    build_explain_with(input, &JsonBuilder::host()).expect("host DTO construction is infallible")
}

pub(crate) fn build_explain_in(
    input: ExplainInput<'_>,
    work: &mut JsonReservation,
) -> Result<JsonValue> {
    // The caller retains credit for the filter tree before handing it over. Charge
    // the new explanation fields to the same result reservation so the completed
    // row can outlive the scoring scratch.
    build_explain_with(
        input,
        &JsonBuilder::budgeted(work, JsonInputKind::StoredJson),
    )
}

fn build_explain_with(input: ExplainInput<'_>, builder: &JsonBuilder<'_>) -> Result<JsonValue> {
    // A route gets credit for recalling a chunk only if its bounded recall window
    // contained the ID. A raw score can also come from score completion, so its
    // presence does not establish candidate provenance.
    let candidate_sources: &[&str] = match (input.bm25_recalled, input.vector_recalled) {
        (true, true) => &["bm25", "vector"],
        (true, false) => &["bm25"],
        (false, true) => &["vector"],
        (false, false) => &[],
    };
    // Build from the saved score and provenance for this hit. Re-running retrieval
    // here could select a different candidate set and would charge another scan to
    // what is only an explanation.
    builder.object([
        ("schema_version", builder.string("0.1")?),
        ("chunk_id", builder.string(&input.chunk.chunk_id)?),
        (
            "chunk_external_id",
            builder.string(&input.chunk.chunk_external_id)?,
        ),
        ("document_id", builder.string(&input.chunk.document_id)?),
        (
            "document_external_id",
            builder.string(input.document_external_id)?,
        ),
        ("collection", builder.string(input.collection_name)?),
        ("final_score", JsonValue::Number(input.final_score)),
        // Expose the raw measurements alongside the quantities fusion used. A
        // Euclidean distance grows as the match gets worse, while its contribution
        // score grows as the match improves. Keeping both makes a low final score
        // interpretable.
        (
            "scores",
            builder.object([
                (
                    "bm25_raw",
                    input
                        .bm25_raw
                        .map(JsonValue::Number)
                        .unwrap_or(JsonValue::Null),
                ),
                ("bm25_norm", JsonValue::Number(input.bm25_norm)),
                (
                    "vector_raw",
                    input
                        .vector_raw
                        .map(JsonValue::Number)
                        .unwrap_or(JsonValue::Null),
                ),
                (
                    "vector_norm",
                    input
                        .vector_norm
                        .map(JsonValue::Number)
                        .unwrap_or(JsonValue::Null),
                ),
                ("metadata_boost", JsonValue::Number(input.metadata_boost)),
                ("final_score", JsonValue::Number(input.final_score)),
            ])?,
        ),
        // The three direct weight fields are aliases of the effective mix used in
        // the sum. The requested mix and source explain how the same request can
        // yield different coefficients with different available inputs.
        (
            "weights",
            builder.object([
                ("bm25", JsonValue::Number(input.weights.effective.bm25)),
                ("vector", JsonValue::Number(input.weights.effective.vector)),
                (
                    "metadata",
                    JsonValue::Number(input.weights.effective.metadata),
                ),
                (
                    "requested",
                    weights_json_in(input.weights.requested, builder)?,
                ),
                (
                    "effective",
                    weights_json_in(input.weights.effective, builder)?,
                ),
                ("source", builder.string(&input.weights.source)?),
                (
                    "renormalization_reason",
                    builder.optional_str(input.weights.renormalization_reason.as_deref())?,
                ),
            ])?,
        ),
        // The normalization population is the deduplicated union after the branch
        // filters and before fusion truncation. A strong lexical score found while
        // completing a vector hit can set the maximum even if that hit falls
        // outside the returned top-k. A zero maximum gives every hit a zero lexical
        // component.
        (
            "normalization",
            builder.object([
                ("scope", builder.string("candidate_set")?),
                (
                    "candidate_count",
                    JsonValue::Number(input.normalization_candidate_count as f64),
                ),
                ("bm25_min", JsonValue::Number(0.0)),
                // Use null when there was no text route. A text route with no
                // positive scores instead reports zero for the observed maximum.
                (
                    "bm25_max",
                    input
                        .bm25_max
                        .map(JsonValue::Number)
                        .unwrap_or(JsonValue::Null),
                ),
                (
                    "vector_method",
                    builder.string(vector_normalization_method(
                        input.distance_metric,
                        input.vectors_normalized,
                    ))?,
                ),
            ])?,
        ),
        // The three term views share the same match list and order. The short list
        // names the matches, the BM25 view supplies the inputs for each raw
        // contribution, and the compact detail view pairs a frequency with its
        // contribution. Emit the same term wherever these representations overlap.
        // Each view owns its own JSON strings, even though they all borrow the
        // same term here. Let the builder charge each instance.
        (
            "matched_terms",
            builder.array(
                input
                    .matched_terms
                    .iter()
                    .map(|term| builder.string(&term.term)),
            )?,
        ),
        // Retrieval indexes chunks as the BM25 documents. Thus tf counts
        // occurrences in this chunk and df counts matching chunks in the
        // collection. The chunk lengths used by the formula are token counts.
        (
            "bm25",
            builder.object([
                ("k1", JsonValue::Number(crate::index::text::BM25_K1)),
                ("b", JsonValue::Number(crate::index::text::BM25_B)),
                (
                    "terms",
                    builder.array(input.matched_terms.iter().map(|term| {
                        builder.object([
                            ("term", builder.string(&term.term)?),
                            ("tf", JsonValue::Number(term.frequency as f64)),
                            ("df", JsonValue::Number(term.document_frequency as f64)),
                            ("chunk_length", JsonValue::Number(term.chunk_length as f64)),
                            ("avg_chunk_length", JsonValue::Number(term.avg_chunk_length)),
                            (
                                "raw_contribution",
                                JsonValue::Number(term.bm25_contribution),
                            ),
                        ])
                    }))?,
                ),
            ])?,
        ),
        (
            "matched_term_details",
            builder.array(input.matched_terms.iter().map(|term| {
                builder.object([
                    ("term", builder.string(&term.term)?),
                    ("frequency", JsonValue::Number(term.frequency as f64)),
                    (
                        "bm25_contribution",
                        JsonValue::Number(term.bm25_contribution),
                    ),
                ])
            }))?,
        ),
        // This describes how the term evidence was tokenized. The query context
        // separately records whether a text route was present, so this object can
        // accompany empty term arrays on a vector-only hit.
        // The decoder currently accepts only the canonical default_en
        // configuration. If collections gain a choice of tokenizer, carry the
        // selected configuration into this explanation.
        (
            "tokenizer",
            crate::TokenizerConfig::default().build_json(builder)?,
        ),
        ("fusion", builder.string("weighted_sum")?),
        (
            "candidate_sources",
            builder.array(
                candidate_sources
                    .iter()
                    .map(|source| builder.string(source)),
            )?,
        ),
        // The profile and stage limits belong to the whole query. A limit is a
        // maximum number of retained candidates, whereas an exact vector scan may
        // have inspected many more entries to select them.
        (
            "resource_profile",
            builder.string(input.resource_profile.as_json_str())?,
        ),
        (
            "candidate_limits",
            builder.object([
                (
                    "bm25",
                    JsonValue::Number(input.candidate_limits.bm25 as f64),
                ),
                (
                    "vector",
                    JsonValue::Number(input.candidate_limits.vector as f64),
                ),
                (
                    "fusion",
                    JsonValue::Number(input.candidate_limits.fusion as f64),
                ),
            ])?,
        ),
        // These are the query's observed totals. Repeating them here makes a single
        // explanation useful on its own. Summing them across returned rows would
        // count the same scan repeatedly.
        (
            "execution_stats",
            builder.object([
                (
                    "vector_scan_count",
                    JsonValue::Number(input.vector_scan_count as f64),
                ),
                ("distance_ops", JsonValue::Number(input.distance_ops as f64)),
                (
                    "fusion_candidate_count",
                    JsonValue::Number(input.fusion_candidate_count as f64),
                ),
            ])?,
        ),
        // Retain the final index source together with the rebuild measurements. A
        // fallback can occur after preflight, so a label derived only from the
        // initially available checkpoint would give the wrong provenance.
        ("index_source", builder.string(input.index_source)?),
        (
            "index_rebuild_reason",
            builder.optional_str(input.index_rebuild_reason)?,
        ),
        (
            "transient_rebuild",
            input.transient_rebuild.build_json(builder)?,
        ),
        // The raw kind and the transform must match the collection's scorer. A raw
        // dot product for vectors that are not declared normalized can be negative
        // or exceed one, and the transform preserves it.
        ("distance_metric", builder.string(input.distance_metric)?),
        (
            "vector_raw_kind",
            builder.string(vector_raw_kind(input.distance_metric))?,
        ),
        (
            "normalization_method",
            builder.string(vector_normalization_method(
                input.distance_metric,
                input.vectors_normalized,
            ))?,
        ),
        // The filter trace belongs to this hit and moves into the explanation. The
        // boost digest and rule count are common to the query, but the number of
        // rules matched is specific to the hit, even when its capped numeric boost
        // coincides with another hit's.
        ("filter", input.filter),
        (
            "metadata_boost_context",
            builder.object([
                ("present", JsonValue::Bool(input.metadata_boost_present)),
                ("enabled", JsonValue::Bool(input.metadata_boost_enabled)),
                (
                    "rule_count",
                    JsonValue::Number(input.boost_rule_count as f64),
                ),
                (
                    "matched_rule_count",
                    JsonValue::Number(input.matched_boost_rule_count as f64),
                ),
                (
                    "config_hash",
                    builder.optional_str(input.boost_config_hash)?,
                ),
            ])?,
        ),
        ("chunk_boundary", {
            // Source offsets and line numbers are separate coordinates. Preserve
            // absent byte offsets as null and add the line pair only for a span that
            // carries line information.
            let span = input.chunk.source_span.as_ref();
            // Source byte ranges use an exclusive end. The writer emits large
            // offsets as decimal strings above JSON's exact integer range, so
            // adjacent offsets remain distinguishable.
            let start = builder.optional_u64(span.and_then(|span| span.start))?;
            let end = builder.optional_u64(span.and_then(|span| span.end))?;
            // Source-span validation pairs the two line numbers. Once a starting
            // line is present, the end is available as well. Both line numbers are
            // one-based and the end line is inclusive.
            if let Some(span) = span.filter(|span| span.line_start.is_some()) {
                builder.object([
                    ("start", start),
                    ("end", end),
                    (
                        "line_start",
                        JsonValue::Number(span.line_start.unwrap() as f64),
                    ),
                    ("line_end", JsonValue::Number(span.line_end.unwrap() as f64)),
                ])?
            } else {
                builder.object([("start", start), ("end", end)])?
            }
        }),
        // Keep the document's supplied URI alongside the display path. The fallback
        // may make these values identical, but each is an independently owned JSON
        // string and must fit the caller's reservation.
        (
            "source",
            builder.object([
                (
                    "document_path",
                    builder.optional_str(document_path(
                        input.document_metadata,
                        input.document_source_uri,
                    ))?,
                ),
                (
                    "source_uri",
                    builder.optional_str(input.document_source_uri)?,
                ),
            ])?,
        ),
    ])
}

pub(crate) fn weights_json_in(
    weights: SearchWeights,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    builder.object([
        ("bm25", JsonValue::Number(weights.bm25)),
        ("vector", JsonValue::Number(weights.vector)),
        ("metadata", JsonValue::Number(weights.metadata)),
    ])
}

fn vector_raw_kind(distance_metric: &str) -> &'static str {
    match distance_metric {
        "dot_product" => "dot_product",
        "euclidean" => "distance",
        _ => "similarity",
    }
}

fn vector_normalization_method(distance_metric: &str, vectors_normalized: bool) -> &'static str {
    // Use the same cases as vector scoring. Cosine and normalized dot product shift
    // and scale to [0, 1], Euclidean uses 1 / (1 + distance), and an unnormalized
    // dot product passes through unchanged.
    match (distance_metric, vectors_normalized) {
        ("cosine", _) => "cosine_to_unit_interval",
        ("dot_product", true) => "normalized_dot_product_cosine_to_unit_interval",
        ("dot_product", false) => "raw_dot_product_no_normalization",
        ("euclidean", _) => "distance_to_inverse_unit_interval",
        _ => "metric_to_unit_interval",
    }
}

pub(crate) fn candidate_limits_json_in(
    limits: CandidateLimits,
    builder: &JsonBuilder<'_>,
) -> Result<JsonValue> {
    builder.object([
        ("bm25", JsonValue::Number(limits.bm25 as f64)),
        ("vector", JsonValue::Number(limits.vector as f64)),
        ("fusion", JsonValue::Number(limits.fusion as f64)),
    ])
}

fn document_path<'a>(
    metadata: &'a JsonValue,
    document_source_uri: Option<&'a str>,
) -> Option<&'a str> {
    // Only a string-valued metadata path supplies a display location. A missing or
    // differently typed path falls back to the recorded source URI. With neither,
    // the report carries a null path.
    metadata
        .get("path")
        .and_then(JsonValue::as_str)
        .or(document_source_uri)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error_detail_value<'a>(err: &'a Error, key: &str) -> Option<&'a str> {
        err.detail()
            .iter()
            .find(|(field, _)| field == key)
            .map(|(_, value)| value.as_str())
    }

    #[test]
    fn search_result_json_uses_public_normalized_score_keys() {
        let result = SearchResult {
            rank: 1,
            chunk_id: "chunk_1".into(),
            chunk_external_id: "doc#1".into(),
            document_id: "doc_1".into(),
            document_external_id: "doc".into(),
            snippet: "wal recovery".into(),
            document_metadata: JsonValue::object(),
            chunk_metadata: JsonValue::object(),
            final_score: 0.7,
            bm25_score: Some(4.2),
            vector_score: Some(0.8),
            metadata_boost: 0.1,
            normalized_scores: NormalizedScores {
                bm25_norm: 0.6,
                vector_norm: Some(0.8),
                metadata_boost: 0.1,
            },
            explain: object_from_pairs([(
                "scores",
                object_from_pairs([
                    ("bm25_norm", JsonValue::Number(0.6)),
                    ("vector_norm", JsonValue::Number(0.8)),
                    ("metadata_boost", JsonValue::Number(0.1)),
                    ("final_score", JsonValue::Number(0.7)),
                ]),
            )]),
        };

        let json = result.to_json();
        let normalized = json
            .get("normalized_scores")
            .and_then(JsonValue::as_object)
            .expect("normalized_scores object");
        assert!(matches!(
            normalized.get("bm25"),
            Some(JsonValue::Number(0.6))
        ));
        assert!(matches!(
            normalized.get("vector"),
            Some(JsonValue::Number(0.8))
        ));
        assert!(matches!(
            normalized.get("metadata_boost"),
            Some(JsonValue::Number(0.1))
        ));
        assert!(!normalized.contains_key("bm25_norm"));
        assert!(!normalized.contains_key("vector_norm"));
    }

    #[test]
    fn effective_weights_errors_include_unavailable_branch_context() {
        let err = effective_weights(
            Some(SearchWeights::new(0.0, 0.0, 1.0).unwrap()),
            true,
            false,
            false,
        )
        .unwrap_err();

        assert_eq!(err.code(), ErrorCode::InvalidSearchWeights);
        assert_eq!(error_detail_value(&err, "field"), Some("weights"));
        assert_eq!(
            error_detail_value(&err, "weights_reason"),
            Some("effective_sum_zero")
        );
        assert_eq!(
            error_detail_value(&err, "reason"),
            Some("effective_sum_zero")
        );
        assert_eq!(error_detail_value(&err, "has_text"), Some("true"));
        assert_eq!(error_detail_value(&err, "has_vector"), Some("false"));
        assert_eq!(
            error_detail_value(&err, "metadata_boost_available"),
            Some("false")
        );
        assert_eq!(error_detail_value(&err, "requested_metadata"), Some("1"));
    }

    #[test]
    fn weight_context_preflight_preserves_ordered_reasons_and_exact_string_credit() {
        use crate::json::{JsonOperationContext, JsonProcessingPolicyV1};
        use crate::storage::RecordReservation;

        let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
        for seed in 0..100 {
            for mask in 0..8 {
                let has_text = mask & 1 != 0;
                let has_vector = mask & 2 != 0;
                let boost = mask & 4 != 0;
                let requested = (seed % 3 != 0).then_some(SearchWeights {
                    bm25: 0.1 + seed as f64 / 1000.0,
                    vector: 0.4,
                    metadata: 0.5 - seed as f64 / 1000.0,
                });
                let raw =
                    requested.unwrap_or_else(|| default_weights_for(has_text, has_vector, boost));
                let source = if requested.is_some() {
                    "user_override"
                } else if boost {
                    "default_with_boost"
                } else {
                    "default_no_boost"
                };
                let mut expected = raw;
                let mut reasons = Vec::new();
                for (available, value, reason) in [
                    (has_text, &mut expected.bm25, "text branch unavailable"),
                    (
                        has_vector,
                        &mut expected.vector,
                        "vector branch unavailable",
                    ),
                    (boost, &mut expected.metadata, "metadata boost unavailable"),
                ] {
                    if !available && *value > 0.0 {
                        *value = 0.0;
                        reasons.push(reason);
                    }
                }
                let sum = expected.bm25 + expected.vector + expected.metadata;
                if sum > 0.0 && (sum - 1.0).abs() > f64::EPSILON {
                    expected.bm25 /= sum;
                    expected.vector /= sum;
                    expected.metadata /= sum;
                    if reasons.is_empty() {
                        reasons.push("weights normalized to sum 1.0");
                    }
                }
                let reason = (!reasons.is_empty()).then(|| reasons.join("; "));
                let bytes = source.len() + reason.as_ref().map_or(0, String::len);
                let context = JsonOperationContext::standalone();
                let mut held = context.workspace_reservation();
                held.grow_to(limit - bytes, limit - bytes, "operation_scratch")
                    .unwrap();
                let mut work =
                    RecordReservation::new(context.reservation(), context.workspace_reservation());
                let mut calls = 0;
                let actual =
                    effective_weights_with(requested, has_text, has_vector, boost, |value| {
                        calls += 1;
                        work.copy_string(value, &context)
                    });
                if sum <= 0.0 {
                    assert_eq!(actual.unwrap_err().code(), ErrorCode::InvalidSearchWeights);
                    assert_eq!(calls, 0, "invalid weights allocate no context strings");
                } else {
                    let actual = actual.unwrap();
                    assert_eq!(actual.source, source);
                    assert_eq!(actual.renormalization_reason, reason);
                    assert_eq!(actual.effective.bm25.to_bits(), expected.bm25.to_bits());
                    assert_eq!(actual.effective.vector.to_bits(), expected.vector.to_bits());
                    assert_eq!(
                        actual.effective.metadata.to_bits(),
                        expected.metadata.to_bits()
                    );
                    assert_eq!(work.parts_mut().1.unwrap().reserved_bytes(), bytes);
                    assert_eq!(
                        held.grow_to(limit - bytes + 1, limit - bytes + 1, "operation_scratch")
                            .unwrap_err()
                            .code(),
                        ErrorCode::ResourceExhausted
                    );
                    drop(actual);
                }
                drop(work);
                held.grow_to(limit, limit, "operation_scratch").unwrap();
                let mut work =
                    RecordReservation::new(context.reservation(), context.workspace_reservation());
                let error =
                    effective_weights_with(requested, has_text, has_vector, boost, |value| {
                        work.copy_string(value, &context)
                    })
                    .unwrap_err();
                assert_eq!(
                    error.code(),
                    if sum <= 0.0 {
                        ErrorCode::InvalidSearchWeights
                    } else {
                        ErrorCode::ResourceExhausted
                    }
                );
                assert_eq!(work.parts_mut().1.unwrap().reserved_bytes(), 0);
            }
        }
    }

    #[test]
    fn owned_result_conversion_preserves_json_and_moves_metadata_allocations() {
        let context = SearchQueryContext::persistent_checkpoint(SearchQueryContextInput {
            text_present: true,
            embedding_present: false,
            filter_present: true,
            weights: WeightContext {
                requested: SearchWeights::default_for(true, false),
                effective: SearchWeights::default_for(true, false),
                source: "default_no_boost".into(),
                renormalization_reason: None,
            },
            metadata_boost_present: false,
            metadata_boost_enabled: false,
            boost_config_hash: None,
            boost_rule_count: 0,
            top_k: 1,
            profile: ResourceProfile::Balanced,
            candidate_limits: CandidateLimits {
                bm25: 10,
                vector: 10,
                fusion: 10,
            },
            cache_budget_mb: 256,
            chunks_scanned: 1,
            bytes_read: Some(0),
        });
        let chunk = test_chunk();
        for seed in 0..100 {
            let metadata = object_from_pairs([
                (
                    "payload",
                    JsonValue::String(format!("\u{4e2d}\u{1f600}\n\\\"{seed}")),
                ),
                (
                    "numbers",
                    JsonValue::Array(vec![
                        JsonValue::Number(-0.0),
                        JsonValue::Number(seed as f64 / 7.0),
                    ]),
                ),
            ]);
            let result = SearchResult {
                rank: seed + 1,
                chunk_id: chunk.chunk_id.clone(),
                chunk_external_id: chunk.chunk_external_id.clone(),
                document_id: chunk.document_id.clone(),
                document_external_id: "doc".into(),
                snippet: "alpha".into(),
                document_metadata: metadata.clone(),
                chunk_metadata: metadata.clone(),
                final_score: 0.5,
                bm25_score: Some(1.0),
                vector_score: None,
                metadata_boost: 0.0,
                normalized_scores: NormalizedScores {
                    bm25_norm: 0.5,
                    vector_norm: None,
                    metadata_boost: 0.0,
                },
                explain: build_explain(test_explain_input(&chunk, &metadata)),
            };
            let document_ptr = result
                .document_metadata
                .get("payload")
                .unwrap()
                .as_str()
                .unwrap()
                .as_ptr();
            let chunk_ptr = result
                .chunk_metadata
                .get("payload")
                .unwrap()
                .as_str()
                .unwrap()
                .as_ptr();
            let results = SearchResults {
                schema_version: "0.1".into(),
                collection: "notes".into(),
                query: Some("alpha".into()),
                top_k: 1,
                profile: ResourceProfile::Balanced,
                query_context: context.clone(),
                results: vec![result],
            };
            let expected = results.to_json().to_compact_string();
            let json_context = crate::json::JsonOperationContext::standalone();
            let mut work = json_context.reservation();
            for result in &results.results {
                for value in [
                    &result.document_metadata,
                    &result.chunk_metadata,
                    &result.explain,
                ] {
                    work.merge(
                        value
                            .reserve_retained_in(JsonInputKind::StoredJson, &json_context)
                            .unwrap(),
                    );
                }
            }
            let query_ptr = results.query.as_ref().unwrap().as_ptr();
            let id_ptr = results.results[0].chunk_id.as_ptr();
            let actual = results
                .build_json(&JsonBuilder::budgeted(&mut work, JsonInputKind::StoredJson))
                .unwrap();
            assert_eq!(actual.to_compact_string(), expected);
            let measured = actual
                .reserve_retained_in(JsonInputKind::StoredJson, &json_context)
                .unwrap();
            assert_eq!(work.reserved_bytes(), measured.reserved_bytes());
            assert_eq!(
                actual.get("query").unwrap().as_str().unwrap().as_ptr(),
                query_ptr
            );
            let item = &actual.get("results").unwrap().as_array().unwrap()[0];
            assert_eq!(
                item.get("chunk_id").unwrap().as_str().unwrap().as_ptr(),
                id_ptr
            );
            assert_eq!(
                item.get("document_metadata")
                    .unwrap()
                    .get("payload")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .as_ptr(),
                document_ptr
            );
            assert_eq!(
                item.get("chunk_metadata")
                    .unwrap()
                    .get("payload")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .as_ptr(),
                chunk_ptr
            );

            let expression_ptr = metadata.get("payload").unwrap().as_str().unwrap().as_ptr();
            let diagnosis = WhyNotDiagnosis {
                schema_version: "0.1".into(),
                diagnosis_scope: "debug_exact".into(),
                debug_full_scan: true,
                resource_profile: ResourceProfile::Balanced,
                query_text: Some(format!("query-{seed}")),
                query_context: context.clone(),
                result: object_from_pairs([
                    (
                        "filter_result",
                        object_from_pairs([("expression", metadata)]),
                    ),
                    ("profile", JsonValue::String("existing override".into())),
                ]),
            };
            let expected = diagnosis.to_json().to_compact_string();
            let query_ptr = diagnosis.query_text.as_ref().unwrap().as_ptr();
            let mut diagnosis_work = diagnosis
                .result
                .reserve_retained_in(JsonInputKind::Filter, &json_context)
                .unwrap();
            // The CLI input lease also covered a query vector/boost, now released.
            diagnosis_work
                .grow(1024 * 1024, JsonInputKind::Filter)
                .unwrap();
            let actual = diagnosis.into_json_in(&mut diagnosis_work).unwrap();
            assert_eq!(actual.to_compact_string(), expected);
            let measured = actual
                .reserve_retained_in(JsonInputKind::StoredJson, &json_context)
                .unwrap();
            assert_eq!(diagnosis_work.reserved_bytes(), measured.reserved_bytes());
            assert_eq!(
                actual
                    .get("query_context")
                    .unwrap()
                    .get("text")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .as_ptr(),
                query_ptr
            );
            assert_eq!(
                actual
                    .get("filter_result")
                    .unwrap()
                    .get("expression")
                    .unwrap()
                    .get("payload")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .as_ptr(),
                expression_ptr
            );
        }
    }

    #[test]
    fn candidate_sources_follow_recall_membership_not_score_sign_or_weight() {
        let chunk = test_chunk();
        let metadata = JsonValue::object();
        for seed in 0..100 {
            for (text, vector, expected) in [
                (false, false, "[]"),
                (true, false, r#"["bm25"]"#),
                (false, true, r#"["vector"]"#),
                (true, true, r#"["bm25","vector"]"#),
            ] {
                let mut input = test_explain_input(&chunk, &metadata);
                input.bm25_recalled = text;
                input.vector_recalled = vector;
                // Complete raw scores do not invent a recall source, even at zero weight.
                input.bm25_raw = Some(seed as f64);
                input.vector_raw = Some(seed as f64 - 50.0);
                input.vector_norm = vector.then_some(0.0);
                input.metadata_boost = 0.5;
                let mut weights = input.weights.clone();
                weights.effective = SearchWeights::new(0.0, 0.0, 1.0).unwrap();
                input.weights = &weights;
                let context = crate::json::JsonOperationContext::standalone();
                let mut work = context.reservation();
                let explain = build_explain_in(input, &mut work).unwrap();
                assert_eq!(
                    explain
                        .get("candidate_sources")
                        .unwrap()
                        .to_compact_string(),
                    expected
                );
                let measured = explain
                    .reserve_retained_in(JsonInputKind::StoredJson, &context)
                    .unwrap();
                assert_eq!(work.reserved_bytes(), measured.reserved_bytes());
            }
        }
    }

    #[test]
    fn explain_weight_aliases_preserve_effective_values_and_tracking() {
        let chunk = test_chunk();
        let metadata = JsonValue::object();
        for seed in 0..100 {
            let has_text = seed % 3 != 0;
            let has_vector = seed % 3 != 1;
            let boost = seed / 3 % 2 == 0;
            let requested = (seed % 4 != 0).then(|| {
                let bm25 = (seed + 1) as f64 / 200.0;
                SearchWeights::new(bm25, 0.25, 0.75 - bm25).unwrap()
            });
            let weights = effective_weights(requested, has_text, has_vector, boost).unwrap();
            let mut input = test_explain_input(&chunk, &metadata);
            input.weights = &weights;
            let context = crate::json::JsonOperationContext::standalone();
            let mut work = context.reservation();
            let explain = build_explain_in(input, &mut work).unwrap();
            let actual = explain.get("weights").unwrap();
            for (field, value) in [
                ("bm25", weights.effective.bm25),
                ("vector", weights.effective.vector),
                ("metadata", weights.effective.metadata),
            ] {
                assert_eq!(
                    actual.get(field),
                    Some(&JsonValue::Number(value)),
                    "seed={seed}"
                );
                assert_eq!(
                    actual.get(field),
                    actual.get("effective").unwrap().get(field)
                );
            }
            assert_eq!(actual.as_object().unwrap().len(), 7);
            assert_eq!(
                actual.get("source").and_then(JsonValue::as_str),
                Some(weights.source.as_str())
            );
            assert_eq!(
                actual.get("requested"),
                Some(&weights_json_in(weights.requested, &JsonBuilder::host()).unwrap())
            );
            assert_eq!(
                actual.get("renormalization_reason"),
                Some(
                    &weights
                        .renormalization_reason
                        .map(JsonValue::String)
                        .unwrap_or(JsonValue::Null)
                )
            );
            let measured = explain
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .unwrap();
            assert_eq!(work.reserved_bytes(), measured.reserved_bytes());
        }
    }

    #[test]
    fn explain_scoring_matches_frozen_requirement_oracles() {
        let chunk = test_chunk();
        let metadata = JsonValue::object();
        let requested = SearchWeights::new(0.45, 0.5, 0.05).unwrap();
        // Requirements 8.9/8.10: fixed expected values, not derived from output weights.
        for (text, vector, boost, expected_weights, expected_score, reason) in [
            (true, true, true, [0.45, 0.5, 0.05], 0.80665, None),
            (
                false,
                true,
                true,
                [0.0, 10.0 / 11.0, 1.0 / 11.0],
                (10.0 * 0.9405 + 0.05) / 11.0,
                Some("text branch unavailable"),
            ),
            (
                true,
                false,
                true,
                [0.9, 0.0, 0.1],
                0.6728,
                Some("vector branch unavailable"),
            ),
            (
                true,
                true,
                false,
                [9.0 / 19.0, 10.0 / 19.0, 0.0],
                (9.0 * 0.742 + 10.0 * 0.9405) / 19.0,
                Some("metadata boost unavailable"),
            ),
        ] {
            let weights = effective_weights(Some(requested), text, vector, boost).unwrap();
            let bm25_hits = if text {
                vec![
                    Bm25Hit {
                        chunk_id: chunk.chunk_id.clone(),
                        score: 7.42,
                        ..Default::default()
                    },
                    Bm25Hit {
                        chunk_id: "normalization-reference".into(),
                        score: 10.0,
                        ..Default::default()
                    },
                ]
            } else {
                Vec::new()
            };
            let vector_hits = if vector {
                vec![crate::index::vector::VectorHit {
                    chunk_id: chunk.chunk_id.clone(),
                    raw_score: 0.881,
                    normalized_score: (0.881 + 1.0) / 2.0,
                }]
            } else {
                Vec::new()
            };
            let scores = crate::index::hybrid::fuse(
                &bm25_hits,
                &vector_hits,
                weights.effective,
                10,
                &crate::json::JsonOperationContext::standalone(),
                |id| crate::metadata::MetadataBoostScore {
                    boost: if boost && id == chunk.chunk_id {
                        0.05
                    } else {
                        0.0
                    },
                    matched_rule_count: usize::from(boost && id == chunk.chunk_id),
                },
            )
            .unwrap();
            let score = scores
                .as_slice()
                .iter()
                .find(|score| score.chunk_id == chunk.chunk_id)
                .unwrap();
            let mut input = test_explain_input(&chunk, &metadata);
            input.weights = &weights;
            input.bm25_raw = score.bm25_raw;
            input.bm25_norm = score.bm25_norm;
            input.bm25_max = text.then_some(10.0);
            input.vector_raw = score.vector_raw;
            input.vector_norm = score.vector_norm;
            input.metadata_boost = score.metadata_boost;
            input.final_score = score.final_score;
            let explain = build_explain(input);
            let weights = explain.get("weights").unwrap();
            for (field, expected) in ["bm25", "vector", "metadata"]
                .into_iter()
                .zip(expected_weights)
            {
                assert!((weights.get(field).unwrap().as_f64().unwrap() - expected).abs() < 1e-12);
                assert_eq!(
                    weights.get(field),
                    weights.get("effective").unwrap().get(field)
                );
            }
            assert_eq!(
                weights.get("source").and_then(JsonValue::as_str),
                Some("user_override")
            );
            assert_eq!(
                weights.get("requested").unwrap().to_compact_string(),
                r#"{"bm25":0.45,"metadata":0.05,"vector":0.5}"#
            );
            assert_eq!(
                weights
                    .get("renormalization_reason")
                    .and_then(JsonValue::as_str),
                reason
            );
            for value in [
                explain.get("final_score").unwrap(),
                explain.get("scores").unwrap().get("final_score").unwrap(),
            ] {
                assert!((value.as_f64().unwrap() - expected_score).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn explain_builder_preserves_fields_and_accounts_for_each_string_copy() {
        let chunk = test_chunk();
        let metadata = object_from_pairs([("path", JsonValue::String("\u{4e2d}/notes".into()))]);
        for seed in 0..100 {
            let context = crate::json::JsonOperationContext::standalone();
            let mut work = context.reservation();
            let terms = (0..seed % 7)
                .map(|i| format!("term-{i}"))
                .collect::<Vec<_>>();
            let details = terms
                .iter()
                .map(|term| MatchedTerm {
                    document_frequency: seed + 1,
                    chunk_length: seed + 3,
                    avg_chunk_length: 10.5,
                    term: term.clone(),
                    frequency: 2,
                    bm25_contribution: 0.75,
                })
                .collect::<Vec<_>>();
            let mut input = test_explain_input(&chunk, &metadata);
            input.matched_terms = &details;
            input.document_source_uri = Some("docs/\u{4e2d}.md");
            input.filter =
                object_from_pairs([("expression", JsonValue::String(format!("filter-{seed}")))]);
            work.merge(
                input
                    .filter
                    .reserve_retained_in(JsonInputKind::Filter, &context)
                    .unwrap(),
            );
            let expected = build_explain(input.clone()).to_compact_string();
            let actual = build_explain_in(input, &mut work).unwrap();
            assert_eq!(actual.to_compact_string(), expected);
            assert_eq!(
                actual.get("matched_terms").unwrap(),
                &JsonValue::Array(terms.iter().cloned().map(JsonValue::String).collect())
            );
            let compact = actual
                .get("matched_term_details")
                .unwrap()
                .as_array()
                .unwrap();
            let detailed = actual
                .get("bm25")
                .unwrap()
                .get("terms")
                .unwrap()
                .as_array()
                .unwrap();
            assert_eq!(compact.len(), terms.len());
            assert_eq!(detailed.len(), terms.len());
            for (i, term) in terms.iter().enumerate() {
                assert_eq!(
                    compact[i].get("term").unwrap().as_str(),
                    Some(term.as_str())
                );
                assert_eq!(compact[i].get("frequency").unwrap().as_f64(), Some(2.0));
                assert_eq!(
                    compact[i].get("bm25_contribution").unwrap().as_f64(),
                    Some(0.75)
                );
                assert_eq!(
                    detailed[i].get("term").unwrap().as_str(),
                    Some(term.as_str())
                );
                assert_eq!(detailed[i].get("tf").unwrap().as_f64(), Some(2.0));
                assert_eq!(
                    detailed[i].get("df").unwrap().as_f64(),
                    Some((seed + 1) as f64)
                );
                assert_eq!(
                    detailed[i].get("chunk_length").unwrap().as_f64(),
                    Some((seed + 3) as f64)
                );
                assert_eq!(
                    detailed[i].get("avg_chunk_length").unwrap().as_f64(),
                    Some(10.5)
                );
                assert_eq!(
                    detailed[i].get("raw_contribution").unwrap().as_f64(),
                    Some(0.75)
                );
            }
            let measured = actual
                .reserve_retained_in(JsonInputKind::StoredJson, &context)
                .unwrap();
            assert_eq!(work.reserved_bytes(), measured.reserved_bytes());
        }
    }

    #[test]
    fn explain_builder_rejects_duplicate_source_strings_under_the_same_budget() {
        let context = crate::json::JsonOperationContext::standalone();
        let kind = JsonInputKind::StoredJson;
        let chunk = test_chunk();
        let metadata = JsonValue::object();
        let source = "x".repeat(4 * 1024 * 1024);
        let pointer = source.as_ptr();
        let mut busy = context.reservation();
        busy.grow(
            crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES - 6 * 1024 * 1024,
            kind,
        )
        .unwrap();
        let mut work = context.reservation();
        let mut input = test_explain_input(&chunk, &metadata);
        input.document_source_uri = Some(&source);
        let error = build_explain_in(input, &mut work).unwrap_err();
        assert_eq!(error.code(), ErrorCode::RecoveryError);
        assert_eq!(
            error_detail_value(&error, "json_input_kind"),
            Some("stored_json")
        );
        assert_eq!(source.as_ptr(), pointer);
        assert_eq!(source.len(), 4 * 1024 * 1024);
        drop(work);
        drop(busy);
        context
            .reservation()
            .grow(crate::json::JsonProcessingPolicyV1::WORK_BUDGET_BYTES, kind)
            .unwrap();
    }

    #[test]
    fn explain_reports_euclidean_distance_normalization() {
        let explain = build_explain(ExplainInput {
            collection_name: "notes",
            chunk: &Chunk {
                chunk_id: "chunk_1".into(),
                document_id: "doc_1".into(),
                collection_id: "col_1".into(),
                chunk_external_id: "doc#1".into(),
                text: "alpha beta".into(),
                text_hash: "hash".into(),
                ordinal: 0,
                token_count: 2,
                metadata: JsonValue::object(),
                embedding_id: Some("emb_1".into()),
                source_span: None,
                created_at: "2026-05-31T00:00:00Z".into(),
            },
            document_external_id: "doc",
            document_metadata: &JsonValue::object(),
            document_source_uri: None,
            bm25_recalled: false,
            vector_recalled: true,
            normalization_candidate_count: 1,
            bm25_raw: None,
            bm25_norm: 0.0,
            bm25_max: None,
            vector_raw: Some(5.0),
            vector_norm: Some(1.0 / 6.0),
            metadata_boost: 0.0,
            final_score: 1.0 / 6.0,
            matched_terms: &[],
            weights: &WeightContext {
                requested: SearchWeights::default_for(false, true),
                effective: SearchWeights::default_for(false, true),
                source: "default".into(),
                renormalization_reason: None,
            },
            resource_profile: ResourceProfile::Balanced,
            candidate_limits: CandidateLimits {
                bm25: 0,
                vector: 10,
                fusion: 10,
            },
            vector_scan_count: 1,
            distance_ops: 1,
            fusion_candidate_count: 1,
            index_source: "exact",
            index_rebuild_reason: None,
            transient_rebuild: TransientRebuildContext {
                started: false,
                completed: false,
                cancelled: false,
                budget_mb: 0,
                chunks_scanned: None,
                bytes_read: None,
                budget_exceeded: false,
            },
            distance_metric: "euclidean",
            vectors_normalized: false,
            filter: JsonValue::object(),
            metadata_boost_present: false,
            metadata_boost_enabled: false,
            boost_config_hash: None,
            boost_rule_count: 0,
            matched_boost_rule_count: 0,
        });

        assert_eq!(
            explain.get("vector_raw_kind").and_then(JsonValue::as_str),
            Some("distance")
        );
        assert_eq!(
            explain
                .get("normalization_method")
                .and_then(JsonValue::as_str),
            Some("distance_to_inverse_unit_interval")
        );
        let normalization = explain
            .get("normalization")
            .and_then(JsonValue::as_object)
            .expect("normalization object");
        assert_eq!(
            normalization
                .get("vector_method")
                .and_then(JsonValue::as_str),
            Some("distance_to_inverse_unit_interval")
        );
    }

    #[test]
    fn dot_product_normalization_method_reports_raw_or_normalized_contract() {
        assert_eq!(vector_raw_kind("dot_product"), "dot_product");
        assert_eq!(
            vector_normalization_method("dot_product", false),
            "raw_dot_product_no_normalization"
        );
        assert_eq!(
            vector_normalization_method("dot_product", true),
            "normalized_dot_product_cosine_to_unit_interval"
        );
    }

    #[test]
    fn explain_source_fields_ignore_non_string_metadata_values() {
        let chunk = test_chunk();
        let explain = build_explain(test_explain_input(
            &chunk,
            &object_from_pairs([
                ("path", JsonValue::Array(Vec::new())),
                (
                    "source_uri",
                    object_from_pairs([("nested", JsonValue::String("value".into()))]),
                ),
            ]),
        ));

        let source = explain
            .get("source")
            .and_then(JsonValue::as_object)
            .expect("source object");
        assert!(matches!(source.get("document_path"), Some(JsonValue::Null)));
        assert!(matches!(source.get("source_uri"), Some(JsonValue::Null)));

        let explain = build_explain(test_explain_input(
            &chunk,
            &object_from_pairs([("source_uri", JsonValue::String("metadata/wal.md".into()))]),
        ));
        let source = explain
            .get("source")
            .and_then(JsonValue::as_object)
            .expect("source object");
        assert!(matches!(source.get("document_path"), Some(JsonValue::Null)));
        assert!(matches!(source.get("source_uri"), Some(JsonValue::Null)));

        let metadata =
            object_from_pairs([("source_uri", JsonValue::String("metadata/wal.md".into()))]);
        let mut input = test_explain_input(&chunk, &metadata);
        input.document_source_uri = Some("docs/wal.md");
        let explain = build_explain(input);
        let source = explain
            .get("source")
            .and_then(JsonValue::as_object)
            .expect("source object");
        assert_eq!(
            source.get("document_path").and_then(JsonValue::as_str),
            Some("docs/wal.md")
        );
        assert_eq!(
            source.get("source_uri").and_then(JsonValue::as_str),
            Some("docs/wal.md")
        );

        let metadata = object_from_pairs([("path", JsonValue::String("metadata/path.md".into()))]);
        let mut input = test_explain_input(&chunk, &metadata);
        input.document_source_uri = Some("docs/wal.md");
        let explain = build_explain(input);
        let source = explain
            .get("source")
            .and_then(JsonValue::as_object)
            .expect("source object");
        assert_eq!(
            source.get("document_path").and_then(JsonValue::as_str),
            Some("metadata/path.md")
        );
        assert_eq!(
            source.get("source_uri").and_then(JsonValue::as_str),
            Some("docs/wal.md")
        );
    }

    fn test_chunk() -> Chunk {
        Chunk {
            chunk_id: "chunk_1".into(),
            document_id: "doc_1".into(),
            collection_id: "col_1".into(),
            chunk_external_id: "doc#1".into(),
            text: "alpha beta".into(),
            text_hash: "hash".into(),
            ordinal: 0,
            token_count: 2,
            metadata: JsonValue::object(),
            embedding_id: Some("emb_1".into()),
            source_span: None,
            created_at: "2026-05-31T00:00:00Z".into(),
        }
    }

    fn test_explain_input<'a>(
        chunk: &'a Chunk,
        document_metadata: &'a JsonValue,
    ) -> ExplainInput<'a> {
        static WEIGHTS: std::sync::OnceLock<WeightContext> = std::sync::OnceLock::new();
        ExplainInput {
            collection_name: "notes",
            chunk,
            document_external_id: "doc",
            document_metadata,
            document_source_uri: None,
            bm25_recalled: false,
            vector_recalled: true,
            normalization_candidate_count: 1,
            bm25_raw: None,
            bm25_norm: 0.0,
            bm25_max: None,
            vector_raw: Some(1.0),
            vector_norm: Some(1.0),
            metadata_boost: 0.0,
            final_score: 1.0,
            matched_terms: &[],
            weights: WEIGHTS.get_or_init(|| WeightContext {
                requested: SearchWeights::default_for(false, true),
                effective: SearchWeights::default_for(false, true),
                source: "default_no_boost".into(),
                renormalization_reason: None,
            }),
            resource_profile: ResourceProfile::Balanced,
            candidate_limits: CandidateLimits {
                bm25: 0,
                vector: 10,
                fusion: 10,
            },
            vector_scan_count: 1,
            distance_ops: 1,
            fusion_candidate_count: 1,
            index_source: "persistent_checkpoint",
            index_rebuild_reason: None,
            transient_rebuild: TransientRebuildContext {
                started: false,
                completed: false,
                cancelled: false,
                budget_mb: 256,
                chunks_scanned: None,
                bytes_read: None,
                budget_exceeded: false,
            },
            distance_metric: "cosine",
            vectors_normalized: false,
            filter: JsonValue::object(),
            metadata_boost_present: false,
            metadata_boost_enabled: false,
            boost_config_hash: None,
            boost_rule_count: 0,
            matched_boost_rule_count: 0,
        }
    }
}
