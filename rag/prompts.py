DMQR_PROMPT = """
### Instruction ###
You will receive a user’s question that requires retrieving relevant content
through internet search to provide an answer. There are now the following
four rewriting methods, General Search Rewriting, Keyword Rewriting, Pseudo-Answer Rewriting, Core Content Extraction. Based on the characteristics of the
query, please select some of the rewriting methods to rewrite the question.

### General Search Rewriting ###
Rewrite the question into a general query for internet search.
### Keyword Rewriting ###
Extract all keywords from the question and separate them with commas, preserving the amount
of information as in the original question.
### Pseudo-Answer Rewriting ###
Generate an answer for the question, and use the answer to match the real answers from the
search engine.
### Core Content Extraction ###
Reduce the amount of information in the original question, only extracting the most core content.
The rewritten query should be more brief than Keyword Rewriting.

### Output Format ###
Each output line should list the selected rewriting method, starting with its
name followed by the rewritten result.
The final line should explain the selection rationale for these methods and the
exclusion of others, beginning with “reason: ”.

### Example ###
Question: Which city was the site where the armistice agreement officially ending World War I
was signed?
Output:
General Search Rewriting: City where World War I armistice agreement was signed
Keyword Rewriting: World War I, Armistice, Signing Location

Begin! Only output the final result without any additional content. Do not
generate any other unrelated content.
Question: {query}
Output:
"""