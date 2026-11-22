//! Text and vector recall produce bounded candidate windows. Hybrid fusion takes
//! their union and combines scores for those candidates; completing a missing
//! component does not introduce another chunk.
pub(crate) mod hybrid;
pub(crate) mod text;
pub(crate) mod vector;
