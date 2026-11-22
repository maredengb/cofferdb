//! Project the ordinary result rows and the target diagnosis into one offline document.
use super::{diagnosis_conclusion, DiagnosisTarget, JsonValue, MaintenanceReceipt};
use crate::error::Result;
use crate::query::SearchResults;
use crate::temporary::{create_temporary_file_in, TempFileGuard};
use std::borrow::Cow;
use std::fmt::{self, Display, Formatter, Write as _};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

// Keep HTML escaping at the point a value is written. Queries, external IDs and
// strings inside displayed JSON can contain markup or literal entity spellings and
// must remain ordinary report text.
struct Text<'a>(&'a str);

impl Display for Text<'_> {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        // Stream the source characters into the formatter. Escaped entity text
        // goes straight to output and is not fed back into this loop, so a literal
        // ampersand is escaped once.
        for c in self.0.chars() {
            match c {
                '&' => out.write_str("&amp;")?,
                '<' => out.write_str("&lt;")?,
                '>' => out.write_str("&gt;")?,
                '"' => out.write_str("&quot;")?,
                '\'' => out.write_str("&#39;")?,
                // Pass other Unicode characters through unchanged. The formatter's
                // first error ends this fragment and propagates to the staging
                // write, before a report is published.
                _ => out.write_char(c)?,
            }
        }
        Ok(())
    }
}

// Only engine-built DTO fields reach this renderer.
// Missing required members indicate a mismatch with the engine's diagnosis schema.
// Optional values still appear as explicit JSON nulls and are rendered by cell.
fn field<'a>(obj: &'a JsonValue, key: &str) -> &'a JsonValue {
    obj.get(key).expect("required diagnostic field")
}

// Keep plain strings readable without surrounding JSON quotes. Use a distinct
// marker for absent values, and preserve the compact JSON representation of
// structured scores and evidence.
fn cell(value: &JsonValue) -> Cow<'_, str> {
    match value {
        JsonValue::String(text) => Cow::Borrowed(text),
        JsonValue::Null => "—".into(),
        _ => Cow::Owned(value.to_compact_string()),
    }
}

// A bare output name has an empty parent component. Use the current directory for
// both temporary-file creation and the directory sync performed after publication.
fn directory(path: &str) -> &Path {
    Path::new(path)
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

pub(super) struct Report {
    // Buffered writer for the staged HTML report, flushed and synced before
    // publication.
    writer: BufWriter<File>,
    // Guard for the temporary name, used to link the completed report into place.
    temp: TempFileGuard,
    // Requested final name. An existing file is protected by new-file publication.
    output: String,
}

impl Report {
    pub(super) fn create(output: &str) -> Result<Self> {
        // Stage beside the requested output so the completed report can be
        // published with a no-replace hard link on the same filesystem. Opening
        // the writer at this stage must not truncate a pre-existing destination.
        let (file, temp) = create_temporary_file_in(directory(output), "explain-report", "tmp")?;
        Ok(Self {
            writer: BufWriter::new(file),
            temp,
            output: output.to_owned(),
        })
    }

    // Write search rows into the staging file before adding the target diagnosis.
    // The holder can release the full search result once this projection is
    // written instead of retaining it through report publication.
    pub(super) fn write_search(&mut self, rows: &SearchResults) -> Result<()> {
        let w = &mut self.writer;
        // The header belongs to the executable at build time. Report generation
        // requires no template lookup relative to the caller's current working
        // directory.
        w.write_all(include_bytes!("report_header.html"))?;
        // Take retrieval mode from the executed query's context. A result with no
        // hits still needs to distinguish a text-only, vector-only or combined
        // query.
        let ctx = &rows.query_context;
        let routes = match (ctx.text_present, ctx.embedding_present) {
            (true, true) => "BM25 and exact vector",
            (true, false) => "BM25",
            _ => "Exact vector",
        };
        // Show the weights actually used in scoring. A query can request weights
        // that are adjusted for the retrieval routes that were available.
        let weights = ctx.weights.effective;
        write!(w, "<dl><dt>Collection</dt><dd>{}</dd><dt>Query</dt><dd>{}</dd><dt>Retrieval</dt><dd>{routes}</dd><dt>Profile</dt><dd>{}</dd><dt>Top-k</dt><dd>{}</dd><dt>Effective weights</dt><dd>BM25 {}, vector {}, metadata {}</dd><dt>Filter</dt><dd><a href='#filter'>View filter evidence</a></dd></dl></section>",
            Text(&rows.collection), Text(rows.query.as_deref().unwrap_or("Vector only")), rows.profile.as_json_str(), rows.top_k, weights.bm25, weights.vector, weights.metadata)?;
        write!(
            w,
            "<section id='results'><h2>Returned results ({})</h2>",
            rows.results.len()
        )?;
        // An empty result set still has its query context and is followed by the
        // target diagnosis. Emit an explicit empty section so absence of hits is
        // not mistaken for an incomplete report.
        if rows.results.is_empty() {
            w.write_all(b"<p>No results returned.</p>")?;
        }
        // Keep the engine's returned order and rank. A report must not reorder
        // equal scores or infer a new rank from the sequence of displayed rows.
        for row in &rows.results {
            let explain = &row.explain;
            write!(w, "<article><h3>Rank {} &middot; {}</h3><dl><dt>Chunk ID</dt><dd>{}</dd><dt>External ID</dt><dd>{}</dd>",
                row.rank, Text(&row.chunk_external_id), Text(&row.chunk_id), Text(&row.chunk_external_id))?;
            // Read the individual score components and candidate sources from the
            // stored explanation for this hit. Recomputing from displayed values
            // could discard route normalization or rounding.
            let scores = field(explain, "scores");
            for (label, value) in [
                ("Final score", field(scores, "final_score")),
                ("Candidate sources", field(explain, "candidate_sources")),
                ("Matched terms", field(explain, "matched_terms")),
                ("BM25 normalized", field(scores, "bm25_norm")),
                ("Vector normalized", field(scores, "vector_norm")),
                ("Metadata boost", field(scores, "metadata_boost")),
            ] {
                // Escape after converting each value to display text. A term or
                // other string inside a JSON array can itself contain HTML
                // characters.
                write!(w, "<dt>{label}</dt><dd>{}</dd>", Text(&cell(value)))?;
            }
            w.write_all(b"</dl></article>")?;
        }
        w.write_all(b"</section>")?;
        Ok(())
    }

    // Append the diagnosis to the staged report and publish the completed file.
    // Keep preparation failures separate from errors after the destination becomes
    // visible.
    pub(super) fn finish(
        mut self,
        result: &JsonValue,
        requested: DiagnosisTarget<'_>,
    ) -> Result<String> {
        // Read all display fields from the completed diagnosis object. Search rows
        // may already be written by write_search, but final publication waits for
        // the target and filter evidence.
        let outcome = field(result, "outcome");
        let filter = field(result, "filter_result");
        let target = field(result, "expected_chunk");
        let w = &mut self.writer;
        write!(
            w,
            "<section id='target'><h2>Target diagnosis</h2><p class='conclusion'>{}</p><dl>",
            // Use the stable diagnosis reason to choose a human-readable
            // conclusion. Keep the original reason visible among the detailed
            // fields below.
            diagnosis_conclusion(field(outcome, "reason").as_str())
        )?;
        // Distinguish the submitted selector from resolved identifiers. The caller
        // can request an internal or external key even when no matching chunk
        // exists.
        let (kind, id) = match requested {
            DiagnosisTarget::Internal(id) => ("Internal ID", id),
            DiagnosisTarget::External(id) => ("External ID", id),
            DiagnosisTarget::Legacy(id) => ("ID", id),
        };
        write!(
            w,
            "<dt>Requested identifier</dt><dd>{}</dd><dt>Identifier type</dt><dd>{kind}</dd>",
            Text(id)
        )?;
        // The legacy JSON echoes the requested key in chunk_id for an absent target.
        // Require a positive existence flag before displaying the internal ID.
        // Older JSON may echo the requested text in this field when lookup found
        // nothing.
        let internal_id = if field(target, "exists").as_bool() == Some(true) {
            field(target, "chunk_id")
        } else {
            &JsonValue::Null
        };
        // Keep candidate and returned ranks distinct from the configured limits. A
        // target with no returned rank can still have a known top-k.
        for (label, value) in [
            ("Chunk ID", internal_id),
            ("External ID", field(target, "chunk_external_id")),
            ("Stage", field(outcome, "stage")),
            ("Reason", field(outcome, "reason")),
            (
                "Fusion rank",
                field(field(result, "candidate_ranks"), "fusion_rank"),
            ),
            (
                "Fusion limit",
                field(field(result, "candidate_limits"), "fusion"),
            ),
            ("Top-k", field(field(result, "query_context"), "top_k")),
            ("Returned rank", field(result, "returned_rank")),
        ] {
            // Escape values from the report data before writing them into HTML.
            // Static labels are supplied by this module and do not pass through
            // the same formatter.
            write!(w, "<dt>{label}</dt><dd>{}</dd>", Text(&cell(value)))?;
        }
        w.write_all(b"</dl><h3>Candidate routes</h3><div class='scroll'><table><thead><tr><th>Route</th><th>Status</th><th>Rank</th><th>Limit</th></tr></thead><tbody>")?;
        // Read route evidence from the diagnosis rather than guessing from the
        // returned rows. A target can miss one route and still have an outcome on
        // the other.
        for (key, label) in [("bm25", "BM25"), ("exact", "Exact vector")] {
            let route = field(outcome, key);
            // Status is required in each route of the internally built diagnosis
            // schema. Optional rank and limit go through the missing-value cell
            // formatter.
            let status = field(route, "status").as_str().unwrap();
            write!(w, "<tr><th scope='row'>{label}</th><td>{} <code>{}</code></td><td>{}</td><td>{}</td></tr>",
                super::route_explanation(status), Text(status), Text(&cell(field(route,"rank"))), Text(&cell(field(route,"limit"))))?;
        }
        // Display the filter expression and its decision alongside individual
        // reasons. The target's filter evidence belongs to this same diagnostic
        // run.
        write!(w, "</tbody></table></div><h3 id='filter'>Filter evidence</h3><dl><dt>Expression</dt><dd>{}</dd><dt>Matched</dt><dd>{}</dd></dl>",
            Text(&cell(field(filter,"expression"))), Text(&cell(field(filter,"matched"))))?;
        for reason in field(filter, "reasons").as_array().unwrap() {
            write!(w, "<p>{}</p>", Text(reason.as_str().unwrap()))?;
        }
        // Complete the staged document before publishing its name. A failed write
        // here leaves the destination untouched, including any report already at
        // that path.
        w.write_all("</section><footer>Ranks start at 1. A dash (—) means no value applies. This local report may contain private queries and identifiers.</footer></main></body></html>".as_bytes())?;
        // Flush the user-space buffer and sync the staged file before linking it.
        // Close the writer first so its destructor cannot make another write after
        // publication.
        w.flush()?;
        w.get_ref().sync_all()?;
        drop(self.writer);
        // Publish via a no-replace hard link. If unlinking the temporary name
        // fails after that point, report the published target and cleanup stage.
        self.temp.link_new(Path::new(&self.output), |err| {
            MaintenanceReceipt::published_report(&self.output)
                .error(err.into(), "cleanup_temporary")
        })?;
        // Sync the containing directory after linking. An error here is a
        // post-publication failure; report the existing target so a retry does not
        // assume the path is still free.
        File::open(directory(&self.output))
            .and_then(|dir| dir.sync_all())
            .map_err(|err| {
                MaintenanceReceipt::published_report(&self.output)
                    .error(err.into(), "sync_directory")
            })?;
        Ok(self.output)
    }
}
