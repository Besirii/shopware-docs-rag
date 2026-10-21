# Udhëzues mësimi: si funksionon ky RAG dhe si ta studiosh

Ky dokument është për ty si zhvillues që do të mësosh RAG duke lexuar një projekt të vërtetë.
README-ja (në anglisht) është përmbledhja për portofolin dhe `docs/DESIGN.md` ka detajet teknike. Këtu shpjegohet *pse* çdo pjesë është ashtu siç është.

## 1. Harta e projektit në 30 sekonda

```
shopware/docs (GitHub)                     pyetja e përdoruesit
        │                                          │
  ingest/fetch_docs.py   klonim i cekët, SHA e     │
        │                fiksuar                   ▼
  ingest/corpus.py       lexim nga git,     app/retrieval.py  dense + BM25 → RRF → reranker
        │                pastrim VitePress         │
  ingest/chunking.py     ndarje sipas titujve      ▼
        │                                   app/generation.py prompt me [1]..[n] → LLM → JSON
  ingest/index.py        embeddings → Qdrant       │
        │                                          ▼
        └──────────────► Qdrant ◄─────────  app/main.py   POST /ask → përgjigje + citime
                                                   ▲
                                eval/run_eval.py ──┘  pyetjet → metrikat → eval/results/
```

## 2. Rendi i leximit të kodit

Lexoji skedarët në këtë rend; secili mbështetet te i mëparshmi.

1. **`app/config.py`** – çdo parametër i sistemit në një vend. Vër re `fingerprint()`:
   çdo rezultat eval-i ruan konfigurimin që e prodhoi.
2. **`ingest/markdown.py`** – funksione të pastra. Mëso pse kodi kërkohet me një parser të vërtetë
   (markdown-it) dhe jo me regex: blloqet ``` brenda listave.
3. **`ingest/corpus.py`** – lexim nga `git cat-file`, jo nga disku. Dy arsye: riprodhueshmëria
   dhe kufiri 260 karaktere i shtigjeve në Windows.
4. **`ingest/chunking.py`** – pjesa më e rëndësishme. Lexo docstring-un në krye, pastaj
   `HeadingChunker.sections → _atoms → _group_sections → _pack`.
5. **`app/embeddings.py`** – vër re prefikset e query/dokumentit dhe truncation-in e qartë.
6. **`ingest/index.py`** – idempotenca: ID = hash i përmbajtjes.
7. **`app/retrieval.py`** – formula RRF dhe pse reranker-i punon vetëm mbi top 30.
8. **`app/generation.py`** + **`app/llm.py`** – prompt-i, JSON-i i strukturuar, validimi i citimeve.
9. **`app/pipeline.py`** + **`app/main.py`** – si lidhen të gjitha, plus pragu "nuk e gjeta".
10. **`eval/`** – `metrics.py` (pa LLM), `judge.py` (LLM-gjykues), `run_eval.py`, `calibrate.py`.

## 3. Konceptet kryesore, shkurt

**Chunking.** Embedding-u i një teksti është një "përmbledhje" numerike. Nëse chunk-u përzien dy
tema, vektori nuk i përfaqëson mirë asnjërën. Prandaj ndajmë sipas titujve (H2/H3) dhe nuk e
presim kurrë një bllok kodi: në Shopware përgjigja shpesh është vetë kodi.

**Prefiksi me breadcrumb.** Chunk-u "Run `bin/console plugin:install`" nuk e di se i përket
"Guides > Plugins > Install". Kur e vendosim këtë zinxhir para tekstit, vektori dhe BM25 e "dinë"
kontekstin.

**Dense vs BM25.** Dense kap kuptimin ("si ndryshoj pamjen e faqes së produktit"). BM25 kap fjalë
të sakta (`ProductPageLoadedEvent`, `theme:compile`). Hybrid i kombinon.

**RRF (Reciprocal Rank Fusion).** `score = Σ 1/(k + rank)`. Përdor vetëm renditjen, prandaj nuk ka
nevojë të krahasosh një score BM25 me një cosine similarity, që janë në shkallë të ndryshme.

**Cross-encoder reranker.** Embedding-u llogarit vektorin e pyetjes dhe të dokumentit veç e veç.
Cross-encoder-i i lexon *bashkë* dhe jep një score më të saktë, por është i ngadaltë. Prandaj
përdoret vetëm mbi 30 kandidatët më të mirë.

**Pragu "nuk e gjeta".** Nëse score-i më i mirë i reranker-it është i ulët, sistemi refuzon pa
thirrur fare LLM-në. Pragu kalibrohet me eval (`python -m eval.calibrate`). Me pak pyetje pa
përgjigje rrezikon overfitting; për këtë skripti raporton edhe saktësinë leave-one-out.

**Citimet e verifikueshme.** LLM-ja sheh vetëm `[1]..[n]`. Backend-i heq çdo `[n]` që nuk
ekziston, kështu që çdo link që sheh përdoruesi i përket një pasazhi që ishte para modelit.

## 4. Metrikat e eval-it: çfarë mat secila

| Metrika | Pyetja që i përgjigjet | Kosto |
|---|---|---|
| Hit@k | A u gjet faqja e duhur në top k? | Falas |
| MRR@10 | Sa lart u gjet? (1/rank) | Falas |
| Refusal accuracy | A thotë "nuk e di" kur duhet, dhe vetëm atëherë? | Falas me prag, me LLM ndryshe |
| Faithfulness | A mbështetet çdo pohim i përgjigjes te konteksti? | LLM-gjykues |
| Answer relevancy | A i përgjigjet përgjigjja pyetjes? | LLM + embeddings |
| Context precision | A janë chunks e dobishme në krye të listës? | LLM-gjykues |
| Context recall | A përmban konteksti gjithë ground truth-in? | LLM-gjykues |

Rregull praktik: ekzekuto metrikat falas (`--no-generate`) pas çdo ndryshimi. Metrikat me
gjykues ekzekutoji kur do të mbyllësh një fazë. Gjykuesi duhet të jetë gjithmonë i njëjti model.

## 5. Gabimet reale që u gjetën gjatë ndërtimit (dhe çfarë mësojmë)

1. **Specifikimi kishte shtigje të vjetra.** `services.xml` → dokumentacioni aktual përdor
   `services.php`; `plugin-fundamentals/listening-to-events.md` nuk ekziston. *Mësim:* ground
   truth nxirret nga korpusi i fiksuar, jo nga kujtesa.
2. **Truncation e heshtur.** Shumica e modeleve të embedding-ut presin në 512 tokens pa asnjë
   paralajmërim. *Mësim:* mat gjatësinë në tokens të modelit dhe raporto sa chunks priten.
3. **Memoria e vëmendjes është kuadratike.** Një chunk prej 5 000 tokens në një batch prej 16
   kërkonte gigabajtë RAM. *Mësim:* rendit sipas gjatësisë dhe kufizo tokens për batch.
4. **Padding.** `encode_batch` mbush çdo tekst deri te më i gjati, prandaj numërimi i
   `len(ids)` jepte gjatësi të gabuara. *Mësim:* numëro `attention_mask`.
5. **Modelet e vogla lokale** kthejnë herë pas here `\n` të shpëtuar dy herë ose blloqe kodi të
   pambyllura. *Mësim:* post-processing i vogël dhe i testuar (`repair_markdown`).
6. **Ollama ka kontekst të vogël si parazgjedhje** dhe e pret prompt-in nga fillimi, pra pikërisht
   dokumentacionin. *Mësim:* vendos gjithmonë `num_ctx`.
7. **Edhe seti i eval-it kishte gabime.** Kontrolli automatik i fushës `evidence` zbuloi që në një
   pyetje ground truth-i përmendte `/api/search/...` që faqja e etiketuar nuk e thotë, dhe në një
   tjetër faqja e saktë ishte tjetër. *Mësim:* testo edhe dataset-in (`tests/test_eval.py`).
8. **Gjykuesi LLM duhet validuar para se t'i besosh.** Një model 4B nxirrte "pohime" nga konteksti
   në vend të përgjigjes. Me `<answer>` në krye dhe udhëzime të qarta problemi u zgjidh.
   *Mësim:* lexo me dorë 3–5 vendime të gjykuesit para ekzekutimit të gjatë.
9. **Hardueri është pjesë e arkitekturës.** Me 16 GB RAM, dy punë me modele njëkohësisht (embedding
   + LLM) e çuan makinën në paging dhe çdo gjë ngeci. *Mësim:* mat RAM-in/VRAM-in dhe planifiko
   punët e rënda njëra pas tjetrës.
10. **Një "rrjedhje memorieje" që s'ishte rrjedhje.** Gjykuesi ngadalësohej nga 8 s në minuta për
    thirrje. `docker top` tregoi `llama-server` me 7 GB RSS; `--help` i tij zbuloi `--cache-ram`
    (parazgjedhje 8 GiB): llama.cpp ruan gjendjen KV të çdo prompt-i të vjetër në RAM. Me mijëra
    prompt-e të ndryshme ai vetëm rritet. `LLAMA_ARG_CACHE_RAM=0` e zgjidhi. *Mësim:* kur diçka
    ngadalësohet me kohën, mat memorien e çdo procesi dhe lexo parametrat e parazgjedhur.

## 6. Çfarë treguan matjet (mësimi më i rëndësishëm)

Specifikimi supozonte që çdo hap (chunking → hybrid → reranker) e përmirëson sistemin. Matjet
treguan diçka më interesante:

- **Chunking sipas titujve ndihmoi** (MRR 0.83 → 0.87) dhe nuk preu asnjë bllok kodi.
- **Hybrid klasik e uli MRR-në** (0.87 → 0.83). Arsyeja: pyetjet janë të formuluara me fjalë të
  tjera, BM25 vetëm është i dobët këtu (0.69) dhe RRF me pesha të barabarta i lejon faqet me fjalë
  të ngjashme, por të gabuara, të ngjiten lart. Me peshë 0.5 për BM25 humbja zhduket dhe asnjë
  pyetje nuk del jashtë top 5.
- **Reranker-i e prishi renditjen** (0.80). Modelet MiniLM/jina/bge janë trajnuar në MS MARCO
  (pyetje web-i) dhe preferojnë tekst që "tingëllon si përgjigje", jo pasazhin me identifikuesin
  e saktë. Por score-i i tyre është i dobishëm si **sinjal konfidence** për "nuk e gjeta".
- **Intervalet e besueshmërisë janë të gjera** (±0.09 për MRR me 42 pyetje). Një ndryshim i vogël
  nuk është provë. Prandaj tabela tregon CI, dhe vendimet merren kur disa metrika pajtohen.

*Mësimi:* mos e merr "best practice" si të vërtetë; ndërto eval-in para optimizimit dhe lëre
të dhënat të vendosin. Edhe një rezultat negativ i matur mirë (reranker-i nuk ndihmon këtu) ka
vlerë në portofol, sepse tregon pjekuri.

## 7. Ushtrime (në rend vështirësie)

1. Shto 10 pyetje nga puna jote reale me Shopware në `eval/questions.jsonl` dhe ekzekuto
   `python -m eval.run_eval --no-generate`. A ndryshon renditja e konfigurimeve?
2. Ndrysho `CHUNK_MAX_TOKENS` në 400 dhe 800, rindërto indeksin (`--recreate`) dhe krahaso MRR.
3. Provo RRF me pesha (`rrf_fuse(..., weights=[1.0, 0.5])`). Pse mund të ndihmojë këtu?
4. Shto metadata filtering: lejo `/ask` të kufizohet te `guides/` ose te `resources/references/adr`.
5. Shto query rewriting: LLM-ja gjeneron 2–3 variante të pyetjes, bën retrieval për secilën dhe i
   bashkon me RRF. Mbaje vetëm nëse eval-i tregon përmirësim.
6. Shto një set pyetjesh në shqip dhe krahaso `granite` (vetëm anglisht) me një model shumëgjuhësh
   (`EMBEDDING_MODEL=...`).

## 8. Komandat që do përdorësh më shpesh

```bash
uv sync                                          # instalo varësitë
docker compose up -d qdrant                      # vetëm Qdrant, për punë lokale
uv run python -m ingest                          # shkarko docs + ndërto të dy indekset
uv run uvicorn app.main:app --reload             # API lokalisht
uv run python -m eval.run_eval --no-generate     # metrikat falas
uv run python -m eval.report --update-docs       # rifresko tabelat te docs/DESIGN.md
uv run pytest                                    # testet
```
