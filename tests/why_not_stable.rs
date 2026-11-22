use cofferdb::{
    parse_json, ChunkInput, CollectionConfig, CollectionHandle, Database, DatabaseConfig,
    DistanceMetric, DocumentInput, EmbeddingInput, JsonValue, VectorIndexConfig,
};
use std::{fs, path::PathBuf, process::Command};

struct Fixture(PathBuf);

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn collection(db: &Database, name: &str) -> CollectionHandle {
    db.create_collection(CollectionConfig {
        name: name.into(),
        embedding_dim: 2,
        distance_metric: DistanceMetric::Cosine,
        embedding_model: Some("m".into()),
        vectors_normalized: false,
        vector_index: VectorIndexConfig::Exact,
        tokenizer: Default::default(),
    })
    .unwrap()
}

fn insert(col: &CollectionHandle, doc: &str, ids: &[String], text: &str) {
    col.insert_document(DocumentInput {
        external_id: doc.into(),
        title: None,
        source_uri: None,
        metadata: JsonValue::object(),
        chunks: ids
            .iter()
            .enumerate()
            .map(|(n, id)| ChunkInput {
                chunk_external_id: id.clone(),
                text: text.into(),
                ordinal: n,
                metadata: parse_json(&format!(r#"{{"n":{n}}}"#)).unwrap(),
                source_span: None,
            })
            .collect(),
    })
    .unwrap();
}

impl Fixture {
    fn command(&self, cmd: &str, args: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_coffer"));
        command
            .current_dir(&self.0)
            .args([cmd, "db.coffer"])
            .args(args);
        command
    }

    fn success(&self, cmd: &str, args: &[&str], json: bool) -> String {
        let mut command = self.command(cmd, args);
        if json {
            command.arg("--json");
        }
        let out = command.output().unwrap();
        assert!(
            out.status.success(),
            "{cmd} {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    }

    // Run the real search and both diagnosis formats for each case.
    fn check(
        &self,
        selector: &str,
        id: &str,
        options: &[&str],
        reason: &str,
        words: &[&str],
    ) -> JsonValue {
        let base = [
            "--collection",
            "one",
            "--profile",
            "constrained",
            "--top-k",
            "1",
        ];
        let mut search = base.to_vec();
        let mut i = 0;
        while i < options.len() {
            if options[i] == "--query" {
                search.insert(0, options[i + 1]);
            } else {
                search.extend_from_slice(&options[i..i + 2]);
            }
            i += 2;
        }
        let ordinary = parse_json(&self.success("search", &search, true)).unwrap();
        let mut args = base.to_vec();
        args.extend([selector, id]);
        args.extend_from_slice(options);
        let value = parse_json(&self.success("why-not", &args, true)).unwrap();
        let result = value.get("result").unwrap();
        let outcome = result.get("outcome").unwrap();
        assert_eq!(
            outcome.get("reason").unwrap().as_str(),
            Some(reason),
            "{args:?}"
        );
        let expected_stage = match reason {
            "returned" => "returned",
            "metadata_filter" => "filtered",
            "no_candidate" => "not_candidate",
            "below_top_k" | "fusion_limit" => "not_returned",
            _ => "unavailable",
        };
        assert_eq!(outcome.get("stage").unwrap().as_str(), Some(expected_stage));
        let target = result
            .get("expected_chunk")
            .unwrap()
            .get("chunk_id")
            .unwrap();
        let hits = ordinary
            .get("result")
            .unwrap()
            .get("results")
            .unwrap()
            .as_array()
            .unwrap();
        let rank = hits
            .iter()
            .position(|r| r.get("chunk_id") == Some(target))
            .map(|i| (i + 1) as f64);
        assert_eq!(result.get("returned_rank").unwrap().as_f64(), rank);
        let human = self.success("why-not", &args, false);
        for word in std::iter::once(&id).chain(words) {
            assert!(
                human.contains(word),
                "{args:?}: expected {word:?} in {human}"
            );
        }
        args.extend(["--output", "report.html"]);
        let receipt = parse_json(&self.success("explain-report", &args, true)).unwrap();
        assert_eq!(
            receipt
                .get("result")
                .unwrap()
                .get("output_path")
                .unwrap()
                .as_str(),
            Some("report.html")
        );
        let report_path = self.0.join("report.html");
        let page = fs::read_to_string(&report_path).unwrap();
        fs::remove_file(report_path).unwrap();
        compare_report(&page, ordinary.get("result").unwrap(), result);
        expect_field(&page, "Requested identifier", &JsonValue::String(id.into()));
        let kind = if selector == "--chunk-id" {
            "Internal ID"
        } else {
            "External ID"
        };
        expect_field(&page, "Identifier type", &JsonValue::String(kind.into()));
        result.clone()
    }
}

// Extract the visible definition-list values, decoding HTML text entities.
fn visible(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&amp;", "&")
}

fn expect_field(section: &str, label: &str, expected: &JsonValue) {
    let opening = format!("<dt>{label}</dt><dd>");
    let value = section
        .split_once(&opening)
        .unwrap_or_else(|| panic!("missing {label}"))
        .1
        .split_once("</dd>")
        .unwrap()
        .0;
    let expected = match expected {
        JsonValue::Null => "—".to_owned(),
        JsonValue::String(s) => s.clone(),
        other => other.to_compact_string(),
    };
    assert_eq!(visible(value), expected, "{label}");
}

fn compare_report(page: &str, search: &JsonValue, diagnosis: &JsonValue) {
    let (search_page, target) = page.split_once("<section id='target'>").unwrap();
    let rows = search.get("results").unwrap().as_array().unwrap();
    assert_eq!(rows.len(), search_page.matches("<article>").count());
    if rows.is_empty() {
        assert!(search_page.contains("No results returned."));
    }
    expect_field(search_page, "Collection", search.get("collection").unwrap());
    expect_field(search_page, "Profile", search.get("profile").unwrap());
    expect_field(search_page, "Top-k", search.get("top_k").unwrap());
    if let Some(query) = search.get("query").filter(|v| v.as_str().is_some()) {
        expect_field(search_page, "Query", query);
    } else {
        assert!(search_page.contains("<dd>Vector only</dd>"));
    }
    for (part, row) in search_page.split("<article>").skip(1).zip(rows) {
        let article = part.split_once("</article>").unwrap().0;
        let rank = row.get("rank").unwrap().to_compact_string();
        assert!(article.starts_with(&format!("<h3>Rank {rank} &middot;")));
        let explain = row.get("explain").unwrap();
        for (label, key) in [
            ("Chunk ID", "chunk_id"),
            ("External ID", "chunk_external_id"),
            ("Final score", "final_score"),
        ] {
            expect_field(article, label, row.get(key).unwrap());
        }
        for (label, key) in [
            ("Candidate sources", "candidate_sources"),
            ("Matched terms", "matched_terms"),
        ] {
            expect_field(article, label, explain.get(key).unwrap());
        }
        for (label, key) in [
            ("BM25 normalized", "bm25_norm"),
            ("Vector normalized", "vector_norm"),
            ("Metadata boost", "metadata_boost"),
        ] {
            expect_field(
                article,
                label,
                explain.get("scores").unwrap().get(key).unwrap(),
            );
        }
    }
    let outcome = diagnosis.get("outcome").unwrap();
    let identity = diagnosis.get("expected_chunk").unwrap();
    let internal_id = if identity.get("exists").unwrap().as_bool() == Some(true) {
        identity.get("chunk_id").unwrap()
    } else {
        &JsonValue::Null
    };
    expect_field(target, "Chunk ID", internal_id);
    for (label, parent, key) in [
        ("External ID", "expected_chunk", "chunk_external_id"),
        ("Stage", "outcome", "stage"),
        ("Reason", "outcome", "reason"),
        ("Fusion rank", "candidate_ranks", "fusion_rank"),
        ("Fusion limit", "candidate_limits", "fusion"),
        ("Top-k", "query_context", "top_k"),
        ("Expression", "filter_result", "expression"),
        ("Matched", "filter_result", "matched"),
    ] {
        expect_field(
            target,
            label,
            diagnosis.get(parent).unwrap().get(key).unwrap(),
        );
    }
    expect_field(
        target,
        "Returned rank",
        diagnosis.get("returned_rank").unwrap(),
    );
    for (key, label) in [("bm25", "BM25"), ("exact", "Exact vector")] {
        let route = outcome.get(key).unwrap();
        let section = target
            .split_once(&format!("scope='row'>{label}</th>"))
            .unwrap()
            .1
            .split_once("</tr>")
            .unwrap()
            .0;
        let numbers = |key| match route.get(key).unwrap() {
            JsonValue::Null => "—".to_owned(),
            v => v.to_compact_string(),
        };
        let status = route.get("status").unwrap().as_str().unwrap();
        assert!(section.ends_with(&format!(
            "<code>{status}</code></td><td>{}</td><td>{}</td>",
            numbers("rank"),
            numbers("limit")
        )));
    }
    for reason in diagnosis
        .get("filter_result")
        .unwrap()
        .get("reasons")
        .unwrap()
        .as_array()
        .unwrap()
    {
        assert!(visible(target).contains(reason.as_str().unwrap()));
    }
}

fn assert_route(result: &JsonValue, key: &str, status: &str, rank: Option<usize>) {
    let route = result.get("outcome").unwrap().get(key).unwrap();
    assert_eq!(route.get("status").unwrap().as_str(), Some(status));
    assert_eq!(
        route.get("rank").unwrap(),
        &rank
            .map(|n| JsonValue::Number(n as f64))
            .unwrap_or(JsonValue::Null)
    );
    let expected_limit = if status == "not_requested" {
        JsonValue::Null
    } else {
        JsonValue::Number(50.0)
    };
    assert_eq!(route.get("limit"), Some(&expected_limit));
}

#[test]
fn why_not_explains_actual_search_stages_in_both_cli_formats() {
    let dir = std::env::temp_dir().join(format!("coffer-why-not-stages-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let fixture = Fixture(dir);
    let path = fixture.0.join("db.coffer");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let col = collection(&db, "one");
    let other = collection(&db, "other");
    let mut lex: Vec<_> = (0..51).map(|i| format!("word-{i}")).collect();
    let mut vecs: Vec<_> = (0..51).map(|i| format!("vec-{i}")).collect();
    insert(&col, "words", &lex, "needle");
    insert(&col, "vectors", &vecs, "hay");
    col.put_embeddings(
        vecs.iter()
            .map(|id| EmbeddingInput {
                chunk_external_id: id.clone(),
                model_name: "m".into(),
                dim: 2,
                vector: vec![1.0, 0.0],
            })
            .collect(),
    )
    .unwrap();
    let internal = |id: &String| col.get_chunk_by_external_id(id).unwrap().chunk_id;
    lex.sort_by_cached_key(internal);
    vecs.sort_by_cached_key(internal);
    let collision = internal(&lex[0]);
    insert(&col, "alias", std::slice::from_ref(&collision), "unrelated");
    insert(&other, "other-doc", &["elsewhere".into()], "needle");
    insert(&col, "deleted-doc", &["gone".into()], "needle");
    let remote_internal = other
        .get_chunk_by_external_id("elsewhere")
        .unwrap()
        .chunk_id;
    insert(
        &col,
        "local-alias",
        std::slice::from_ref(&remote_internal),
        "distinct",
    );
    let gone = internal(&"gone".into());
    col.delete_chunk(&gone).unwrap();
    db.close().unwrap();
    let before = fs::read(&path).unwrap();
    fs::write(
        fixture.0.join("query.json"),
        r#"{"model_name":"m","dim":2,"vector":[1,0]}"#,
    )
    .unwrap();
    let key = "--chunk-external-id";
    let word = ["--query", "needle"];
    let vector = ["--embedding", "query.json"];
    for (route, label, opts, ids, inactive) in [
        ("bm25", "BM25", &word, &lex, "exact"),
        ("exact", "Exact vector", &vector, &vecs, "bm25"),
    ] {
        let filtered = [opts[0], opts[1], "--filter", r#"{"chunk.n":-1}"#];
        let r = fixture.check(
            key,
            &ids[0],
            &filtered,
            "metadata_filter",
            &[label, "excluded", "chunk.n"],
        );
        assert_route(&r, route, "filtered", None);
        assert_route(&r, inactive, "not_requested", None);
        assert_eq!(
            r.get("filter_result")
                .unwrap()
                .get("matched")
                .unwrap()
                .as_bool(),
            Some(false)
        );
        for (i, reason, status, expected) in [
            (0, "returned", "candidate", "Returned rank 1"),
            (
                1,
                "below_top_k",
                "candidate",
                "Fusion rank 2, fusion limit 50, top-k 1",
            ),
            (49, "below_top_k", "candidate", "rank 50, limit 50"),
            (50, "no_candidate", "outside_limit", "rank 51, limit 50"),
        ] {
            let r = fixture.check(key, &ids[i], opts, reason, &[label, expected]);
            assert_route(&r, route, status, Some(i + 1));
        }
    }
    let n = lex[50].strip_prefix("word-").unwrap();
    let filter = format!(r#"{{"chunk.n":{n}}}"#);
    let r = fixture.check(
        key,
        &lex[50],
        &["--query", "needle", "--filter", &filter],
        "no_candidate",
        &["rank 51, limit 50"],
    );
    assert_route(&r, "bm25", "outside_limit", Some(51));
    assert_eq!(
        r.get("candidate_ranks")
            .unwrap()
            .get("bm25_rank")
            .unwrap()
            .as_f64(),
        Some(1.0)
    );
    let r = fixture.check(key, &lex[0], &vector, "no_candidate", &["has no embedding"]);
    assert_route(&r, "exact", "missing_embedding", None);
    let r = fixture.check(
        key,
        &vecs[0],
        &word,
        "no_candidate",
        &["no matching query terms"],
    );
    assert_route(&r, "bm25", "no_match", None);
    let hybrid = [
        "--query",
        "needle",
        "--embedding",
        "query.json",
        "--weights",
        r#"{"bm25":1,"vector":0,"metadata":0}"#,
    ];
    let r = fixture.check(
        key,
        &vecs[0],
        &hybrid,
        "fusion_limit",
        &[
            "outside the fusion window",
            "Fusion rank 51, fusion limit 50, top-k 1",
        ],
    );
    assert_route(&r, "exact", "candidate", Some(1));
    assert_route(&r, "bm25", "no_match", None);
    fixture.check(
        key,
        &lex[49],
        &hybrid,
        "below_top_k",
        &["Fusion rank 50, fusion limit 50"],
    );
    for (id, reason, words) in [
        ("missing", "not_found", "not found"),
        (&gone, "deleted", "deleted"),
        ("elsewhere", "collection_mismatch", "another collection"),
    ] {
        fixture.check(key, id, &word, reason, &[words]);
    }
    fixture.check(
        "--chunk-id",
        &collision,
        &word,
        "returned",
        &["Returned rank 1"],
    );
    fixture.check(
        key,
        &collision,
        &word,
        "no_candidate",
        &["no matching query terms"],
    );
    fixture.check(
        "--chunk-id",
        &remote_internal,
        &word,
        "collection_mismatch",
        &["another collection"],
    );
    fixture.check(
        key,
        &remote_internal,
        &word,
        "no_candidate",
        &["no matching query terms"],
    );
    let repeat = fixture.check(key, &vecs[0], &hybrid, "fusion_limit", &["Fusion rank 51"]);
    assert_eq!(r.get("outcome"), repeat.get("outcome"));
    for (file, k) in [("db.coffer", "0"), ("missing.coffer", "1")] {
        for json in [false, true] {
            let mut cmd = Command::new(env!("CARGO_BIN_EXE_coffer"));
            cmd.current_dir(&fixture.0).args([
                "why-not",
                file,
                "--collection",
                "one",
                key,
                "missing",
                "--query",
                "needle",
                "--top-k",
                k,
            ]);
            if json {
                cmd.arg("--json");
            }
            let out = cmd.output().unwrap();
            assert!(!out.status.success() && out.stdout.is_empty() && !out.stderr.is_empty());
            if json {
                assert_eq!(
                    parse_json(std::str::from_utf8(&out.stderr).unwrap())
                        .unwrap()
                        .get("ok")
                        .unwrap()
                        .as_bool(),
                    Some(false)
                );
            }
        }
    }
    assert_eq!(before, fs::read(path).unwrap());
}

#[test]
fn report_preserves_files_and_only_publishes_complete_html() {
    let dir = std::env::temp_dir().join(format!("coffer-report-files-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let fixture = Fixture(dir);
    let path = fixture.0.join("db.coffer");
    let db = Database::create(&path, DatabaseConfig::default()).unwrap();
    let col = collection(&db, "one");
    let hostile = r#"row <img src="https://example.invalid/icon"> & '雪'"#;
    insert(&col, "words", &[hostile.into(), "next".into()], "needle");
    db.close().unwrap();
    let before = fs::read(&path).unwrap();
    let opts = [
        "--collection",
        "one",
        "--query",
        "needle",
        "--chunk-external-id",
        hostile,
    ];
    let report = |name: &str| {
        let mut c = fixture.command("explain-report", &opts);
        c.args(["--output", name, "--json"]);
        c
    };
    let mut jobs: Vec<_> = (0..2)
        .map(|_| {
            use std::process::Stdio;
            report("local.html")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap()
        })
        .collect();
    let outcomes: Vec<_> = jobs
        .drain(..)
        .map(|p| p.wait_with_output().unwrap())
        .collect();
    assert_eq!(outcomes.iter().filter(|x| x.status.success()).count(), 1);
    let page = fs::read_to_string(fixture.0.join("local.html")).unwrap();
    assert!(page
        .contains("&lt;img src=&quot;https://example.invalid/icon&quot;&gt; &amp; &#39;雪&#39;"));
    assert!(!page.contains("<img") && !page.contains("<script"));
    let ordinary =
        parse_json(&fixture.success("search", &["needle", "--collection", "one"], true)).unwrap();
    let why = parse_json(&fixture.success("why-not", &opts, true)).unwrap();
    compare_report(
        &page,
        ordinary.get("result").unwrap(),
        why.get("result").unwrap(),
    );
    assert_eq!(page.matches("<article>").count(), 2);
    for target in ["db.coffer", "local.html", "missing/out.html"] {
        let out = report(target).output().unwrap();
        assert!(!out.status.success() && out.stdout.is_empty() && !out.stderr.is_empty());
    }
    let failed = fixture
        .command(
            "explain-report",
            &[
                "--collection",
                "absent",
                "--query",
                "needle",
                "--chunk-id",
                hostile,
                "--output",
                "failed.html",
                "--json",
            ],
        )
        .output()
        .unwrap();
    assert!(!failed.status.success() && failed.stdout.is_empty() && !failed.stderr.is_empty());
    assert!(!fixture.0.join("failed.html").exists());
    assert!(!fs::read_dir(&fixture.0).unwrap().any(|e| e
        .unwrap()
        .file_name()
        .to_string_lossy()
        .ends_with(".tmp")));
    assert_eq!(
        fs::read_to_string(fixture.0.join("local.html")).unwrap(),
        page
    );
    #[cfg(unix)]
    {
        use std::{
            os::{fd::OwnedFd, unix::net::UnixStream},
            process::Stdio,
        };
        let (sender, reader) = UnixStream::pair().unwrap();
        drop(reader);
        let out = report("delivered.html")
            .stdout(Stdio::from(OwnedFd::from(sender)))
            .output()
            .unwrap();
        assert!(!out.status.success());
        let error = parse_json(std::str::from_utf8(&out.stderr).unwrap()).unwrap();
        let detail = error.get("error").unwrap().get("detail").unwrap();
        assert_eq!(
            detail.get("effect").unwrap().as_str(),
            Some("report_published")
        );
        assert_eq!(
            detail.get("target").unwrap().as_str(),
            Some("delivered.html")
        );
        assert_eq!(
            fs::read_to_string(fixture.0.join("delivered.html")).unwrap(),
            page
        );
    }
    assert_eq!(before, fs::read(path).unwrap());
}
