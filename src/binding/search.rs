use super::{Argument, Operation, Output};
use crate::{
    Database, JsonValue, MetadataBoostConfig, QueryEmbeddingInput, ResourceProfile, Result,
    SearchWeights,
};

/// Owned core query inputs. The optional boost business root releases its raw
/// input before execution while its compiled core representation stays charged.
pub struct SearchInput {
    pub text: Option<String>,
    pub embedding: Option<QueryEmbeddingInput>,
    pub filter: JsonValue,
    pub weights: Option<SearchWeights>,
    pub metadata_boost: Option<Argument<JsonValue>>,
    pub top_k: usize,
    pub profile: Option<ResourceProfile>,
}

impl Operation {
    pub fn weights(&self, value: Argument<JsonValue>) -> Result<SearchWeights> {
        self.check_input(&value)?;
        SearchWeights::from_validated_json(&value.value)
    }

    pub fn search(
        self,
        database: &Database,
        collection_id: &str,
        input: Argument<SearchInput>,
    ) -> Result<Output> {
        self.check_input(&input)?;
        crate::query::validate_search_top_k(input.value.top_k)?;
        if let Some(boost) = &input.value.metadata_boost {
            self.check_input(boost)?;
        }
        let (results, mut work) = {
            let Argument {
                value: payload,
                work: _input,
            } = input;
            let input = payload;
            let boost = input
                .metadata_boost
                .map(|raw| {
                    let Argument {
                        value: payload,
                        work: _raw_work,
                    } = raw;
                    let value = payload;
                    MetadataBoostConfig::parse_validated_in(&value, &self.context)
                })
                .transpose()?;
            database.search_in_collection(
                collection_id,
                input.text.as_deref(),
                input.embedding.as_ref(),
                &input.filter,
                input.weights,
                boost.is_some(),
                boost.as_ref().map(|(value, _)| value),
                input.top_k,
                input.profile,
                Some(self.context.clone()),
            )?
        };
        let value = results.into_json_retained(&mut work)?;
        Ok(Output::new(value, work, Some(database.clone())))
    }
}
