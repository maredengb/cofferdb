use super::{Argument, Operation, Output};
use crate::{
    Database, JsonValue, MetadataBoostConfig, QueryEmbeddingInput, ResourceProfile, Result,
    SearchWeights,
};

/// Owned core query inputs. The optional boost business root releases its raw
/// input before execution while its compiled core representation stays charged.
pub struct SearchInput {
    /// Owned lexical query text. The engine trims it and can still run a
    /// vector-only search when it is blank.
    pub text: Option<String>,
    /// Owned vector, declared dimension and optional model name checked against the
    /// native collection on execution.
    pub embedding: Option<QueryEmbeddingInput>,
    /// Host filter converted to JSON and held with this input. The default empty
    /// object leaves the search unrestricted.
    pub filter: JsonValue,
    /// Optional validated weights. Absence lets the engine choose its defaults
    /// after it knows the available branches.
    pub weights: Option<SearchWeights>,
    /// Raw rule object with its own admission credit, released after conversion
    /// into the separately retained rule set.
    pub metadata_boost: Option<Argument<JsonValue>>,
    /// Maximum returned hits. The shared engine validates its exact JSON integer
    /// range before starting the search.
    pub top_k: usize,
    /// Optional resource preset for this search. An absent override uses the
    /// database's current setting.
    pub profile: Option<ResourceProfile>,
}

impl Operation {
    /// Parse a weights object prepared by this operation into the engine's typed
    /// scoring weights.
    ///
    /// # Errors
    ///
    /// Returns an error for an argument prepared elsewhere or an invalid weight
    /// object. The search request determines which retrieval channels can use the
    /// weights.
    pub fn weights(&self, value: Argument<JsonValue>) -> Result<SearchWeights> {
        self.check_input(&value)?;
        SearchWeights::from_validated_json(&value.value)
    }

    /// Run the prepared search against an internal collection ID and return the
    /// retained result for host conversion.
    ///
    /// # Errors
    ///
    /// Returns an error if the search or its boost was prepared by a different
    /// operation, `top_k` or another query parameter is invalid, or execution or
    /// result projection fails.
    pub fn search(
        self,
        database: &Database,
        collection_id: &str,
        input: Argument<SearchInput>,
    ) -> Result<Output> {
        // Check the outer argument and requested result bound before allocating
        // compiled boost rules.
        self.check_input(&input)?;
        crate::query::validate_search_top_k(input.value.top_k)?;
        // The raw boost is an owned Argument with its own reservation. Check its
        // operation identity separately from the outer query.
        if let Some(boost) = &input.value.metadata_boost {
            self.check_input(boost)?;
        }
        // Keep the outer argument's existing reservation live for as long as the
        // engine borrows its filter, text and vector. Their owned storage need not
        // be copied into another request.
        let (results, mut work) = {
            let Argument {
                value: payload,
                work: _input,
            } = input;
            let input = payload;
            // The raw boost tree is needed only to construct the validated rules.
            // Its guard drops after parsing; the parsed rules and the other input
            // allocations must remain charged through scoring.
            let boost = input
                .metadata_boost
                .map(|raw| {
                    // Move the raw boost and its reservation into this short
                    // parsing scope. The parser returns the compiled configuration
                    // with its own credit, allowing the raw tree to drop before
                    // scoring.
                    let Argument {
                        value: payload,
                        work: _raw_work,
                    } = raw;
                    let value = payload;
                    MetadataBoostConfig::parse_validated_in(&value, &self.context)
                })
                // Propagate a malformed supplied boost as a query error. Absence
                // alone bypasses parsing; a parser failure must not fall back to a
                // search with boosting disabled.
                .transpose()?;
            database.search_in_collection(
                collection_id,
                input.text.as_deref(),
                input.embedding.as_ref(),
                &input.filter,
                input.weights,
                // A supplied configuration requests boosting. The engine records
                // presence separately and enables scoring only when there are
                // rules to apply.
                boost.is_some(),
                boost.as_ref().map(|(value, _)| value),
                input.top_k,
                input.profile,
                // Use this call's operation context for engine execution, keeping
                // admission for the search and the prepared host inputs on the
                // same ledger.
                Some(self.context.clone()),
            )?
        };
        // Only owned results and their existing credit leave the input scope. Build
        // the host JSON after the raw inputs and compiled boost rules have been
        // dropped.
        let value = results.into_json_retained(&mut work)?;
        // Attach the database to the projected result. Materializing a host copy
        // may reclaim optional cache memory after the search and input
        // reservations have gone out of scope.
        Ok(Output::new(value, work, Some(database.clone())))
    }
}
