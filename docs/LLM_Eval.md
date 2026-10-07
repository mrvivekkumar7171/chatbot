### How to Evaluate LLM?

Most people to vibe testing means casually trying an LLM app with few prompts and judging it by feel. This is informal, subjective, and usually not repeatable. We cannot deploy an LLM project in Production on the basis of this. It become difficult to evaluate LLMs because they are not deterministic and can produce different outputs for the same input. This is where LLM evaluation comes into play.

LLM evaluations are structured systematic repeatable tests used to judge an LLM or LLM-powered application against a set of clear criteria. It is not just a metric, but complete testing setup. There can be multiple evaluation pipelines in a system, at component and system level.
- Systematic: We create a `dataset` of different cases and use it to evaluate.
- Repeatable: We should be able to `test` and `compare result` evenafter changes in LLM like prompt, model, retrieval.

**We have to do evaluation in multiple dimensions based on different failure points:**
1. **Components Failure**
- Prompt
- Retrieval
- Reranker
- Query Rewriter
- Embedding Model
- Vector Database
- Output Parser
- Tool Selector
- Memory
- Guardrails
2. **Workflows Failure**
- RAG workflow: like relevent context, retrieval recall, Groundness/faithfulness (claim is supported by the retrived document), citation accuracy.
- Agent workflow: Tool selection, Parameter correctness, Task completion, error recovery etc.
- Multi-turn conversations workflow: context/memory retention, clarification behavior when request is imbageous.
- Structured O/P workflow
- Document Extraction workflow
3. **Entire Application Failure**
- Latency, Cost, fast, token efficiency, cost per request, error/failure rate and reliability like don't crash or fail for scale.
- version 2 is better than version 1?
- do we can use the model for a application in production?
4. **Risk Categories**
- Factual or not.
- safe and policy compliant
- Completeness, Relevant, Groundness, Tonuality, Instruction/Constraints Following like format and length etc.
- Hallucination
- Jail breaking/ Prompt Injection resistance
- No Bias/Fairness againt any group, don't leak private data, no getting tricked into breaking rules, Hurting Anyone, Exposing Anything it shouldn't, No Toxic, Harmful and Dangerous content

### LLM Evaluation
1. **LLM Evaluation**
    - **LLM Model Evaluation**: Benchmark Metrics like Reasoning (MMLU), Knowledge, Maths (GSM8K), Coding (HumanEval, SWE-bench), Instruction Following (IFEval), Long Context (Needle-in-a-Haystack), Multimodel understanding (MMMU), Tool-use.
    - **LLM Application Evaluation**: This is what we do.
2. **LLM Evaluation Pipelines**
3. **RAG Evaluation**
4. **Agent Evaluation**
5. **Safety Evaluation**
6. **Operational Evaluation**

### **LLM Application Evaluation Pipelines**
1. Define Task & Target
2. Define a Success Criteria
3. Build a Dataset from past with actual user queries and expected outputs
4. Define an Evaluation method
5. Run the model
6. Evaluate the results: Using human, automated or another LLM (Best to use LLM like Jev) to calculate the accuracy.
7. Analyze the results:
8. Improve the system: If Improvement needed then start again from step `5`.
8. Deploy & Monitor:
9. Production Failures: Again start from step `3` Build an new dataset.
[LLM Evaluation](/docs/img/LLM_Eval.png)