| Method                                | Family       | Prompt / Retrieval                                               | Vote | Overall Acc | Overall F1 |  LRL Acc |  MRL Acc |  HRL Acc | Status           |
| ------------------------------------- | ------------ | ---------------------------------------------------------------- | ---- | ----------: | ---------: | -------: | -------: | -------: | ---------------- |
| `mean`                                | baseline     | global mean                                                      | no   |    `0.8409` |   `0.7334` | `0.8097` | `0.8719` | `0.8410` | completed        |
| `random`                              | baseline     | random baseline                                                  | no   |    `0.7741` |   `0.6312` | `0.7398` | `0.8124` | `0.7703` | completed        |
| `knn`                                 | baseline     | kNN                                                              | no   |    `0.8631` |   `0.7657` | `0.7555` | `0.9328` | `0.9010` | completed        |
| `softimpute`                          | baseline     | SoftImpute                                                       | no   |    `0.8848` |   `0.8125` | `0.8097` | `0.9413` | `0.9033` | completed        |
| `v4_8b_novote`                        | rag          | `v4_strict_json`                                                 | no   |    `0.8771` |   `0.8088` | `0.9051` | `0.8885` | `0.8379` | completed        |
| `v4_8b_vote`                          | rag          | `v4_strict_json`                                                 | yes  |    `0.8906` |   `0.8234` | `0.9283` | `0.8992` | `0.8442` | completed        |
| `v4_rag_only_json`                    | rag ablation | flat retrieval-only                                              | no   |       `N/A` |      `N/A` |    `N/A` |    `N/A` |    `N/A` | not run          |
| `v4_rag_vote_json`                    | rag ablation | flat retrieval-only                                              | yes  |    `0.8749` |   `0.8048` | `0.9010` | `0.8903` | `0.8334` | completed        |
| `kg_flat`                             | kg           | `v5_glottolog_tree_json` + `kg_flat`                             | yes  |    `0.8834` |   `0.8172` | `0.9145` | `0.8939` | `0.8419` | completed        |
| `kg_typed`                            | kg           | `v5_glottolog_tree_json` + `kg_typed`                            | yes  |    `0.8834` |   `0.8156` | `0.9145` | `0.8966` | `0.8392` | completed        |
| `kg_typed_contrastive`                | kg           | `v5_glottolog_tree_json` + `kg_typed_contrastive`                | yes  |    `0.8849` |   `0.8169` | `0.9158` | `0.8961` | `0.8428` | best KG baseline |
| `kg_typed_compact_fixed`              | kg           | `v5_glottolog_tree_compact_json` + `kg_typed`                    | yes  |    `0.8803` |   `0.8116` | `0.9167` | `0.8894` | `0.8348` | completed        |
| `hybrid_flat_kg_fixed`                | kg hybrid    | `v5_glottolog_tree_json` + `hybrid_flat_kg`                      | yes  |    `0.8686` |   `0.7926` | `0.9140` | `0.8621` | `0.8298` | completed        |
| `kg_flat_retrieval_only`              | kg ablation  | `v5_glottolog_tree_retrieval_only_json` + `kg_flat`              | yes  |    `0.8555` |   `0.7858` | `0.8809` | `0.8670` | `0.8186` | completed        |
| `kg_typed_contrastive_retrieval_only` | kg ablation  | `v5_glottolog_tree_retrieval_only_json` + `kg_typed_contrastive` | yes  |    `0.8601` |   `0.7882` | `0.8679` | `0.8809` | `0.8316` | completed        |

**Notes by run**

- `mean`
  Uses the global training positive rate as a constant prediction rule. No retrieval, no model reasoning, no voting.

- `random`
  Random baseline. No retrieval and no structured evidence.

- `knn`
  Classical non-LLM baseline using nearest-neighbor similarity in the typology table, without LLM prompting.

- `softimpute`
  Classical matrix-completion baseline. This is not a retrieval method and does not use prompting.

- `v4_8b_novote`
  Legacy flat RAG prompt with flat phylogenetic and geographic neighbor retrieval. Includes anchor facts, nearest `1/0` support summaries, correlated clues, and a weak prevalence prior, but no vote table.

- `v4_8b_vote`
  Same as `v4_8b_novote`, but adds the vote table. This is the strongest flat legacy RAG variant.

- `v4_rag_only_json`
  Stripped flat-RAG ablation designed to keep only flat retrieved phylogenetic and geographic evidence. Removes anchors, nearest `1/0`, correlated clues, prevalence prior, and vote.

- `v4_rag_vote_json`
  Same as `v4_rag_only_json`, but adds the vote table. This is the clean flat retrieval-only comparator with vote.

- `kg_flat`
  KG-backed flat retrieval using the standard KG prompt scaffold. Keeps anchors, nearest `1/0`, correlated clues, prevalence prior, and vote. Closest KG analogue to the legacy flat RAG setup.

- `kg_typed`
  Adds typed KG retrieval and graph-aware ranking signals such as relation type, target-feature availability, and structural distance, while keeping the full KG prompt scaffold.

- `kg_typed_contrastive`
  Builds on `kg_typed` by explicitly surfacing support for both competing values `1` and `0`. This is the best-performing full KG baseline.

- `kg_typed_compact_fixed`
  Uses the same typed KG retrieval as `kg_typed`, but with a more compressed prompt serialization.

- `hybrid_flat_kg_fixed`
  Mixes legacy flat retrieval and KG evidence in the same prompt.

- `kg_flat_retrieval_only`
  KG-flat retrieval under a stripped retrieval-only scaffold. Removes anchors, nearest `1/0`, correlated clues, and prevalence prior, leaving KG-flat retrieved evidence plus vote.

- `kg_typed_contrastive_retrieval_only`
  Typed-contrastive KG retrieval under the same stripped retrieval-only scaffold. This is the cleanest current test of typed KG retrieval without the extra prompt-side helpers.

**Key comparison axes**

- `vote` vs `novote`
  Isolates the contribution of the vote table.

- `v4_8b_*` vs `v4_rag_*`
  Isolates how much the legacy flat RAG system benefits from extra prompt-side helpers beyond retrieval.

- `kg_flat` vs `kg_typed`
  Tests whether typed KG ranking helps beyond flat KG retrieval.

- `kg_typed` vs `kg_typed_contrastive`
  Tests whether explicit contrastive support for both values helps beyond typed KG retrieval alone.

- `v4_rag_vote_json` vs `kg_flat_retrieval_only`
  Tests whether KG-backed flat retrieval helps beyond a flat RAG retrieval-only baseline.

- `kg_flat_retrieval_only` vs `kg_typed_contrastive_retrieval_only`
  Tests whether typed/contrastive KG retrieval helps beyond flat KG retrieval under a stripped scaffold.

**Current interpretation**

- The full KG prompts are competitive, but `v4_8b_vote` remains the strongest overall run on this split.
- `v4_rag_vote_json` outperforms both retrieval-only KG variants, so the current evidence does not support a claim that KG retrieval alone beats flat RAG.
- `kg_typed_contrastive_retrieval_only` does outperform `kg_flat_retrieval_only`, which supports a smaller claim that typed KG retrieval adds value over flat KG retrieval under a stripped prompt scaffold.
- `LRL`, `MRL`, and `HRL` are the low-, medium-, and high-resource splits in `coverage_bottomk200_equal2233_v1`.
