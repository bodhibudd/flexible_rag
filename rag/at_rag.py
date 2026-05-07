#source code: https://github.com/MrRezaeiUofT/AT-RAG,略作了修改
from decouple import config
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from typing import List
import time
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict
# Append the necessary paths for dataset and vector DB


class CoTSelfRAG:
    def __init__(
        self,
        max_iter=5,
        max_doc_retrived=5,
    ):
        # Load OpenAI API key from environment variables
        self.openai_api_key = config("OPENAI_API_KEY")
        self.max_iter = max_iter
        self.max_doc_retrived = max_doc_retrived

        # Initialize the Ingestor
        self.llm=ChatOpenAI(model="qwen-max",openai_api_key= self.openai_api_key,
                   openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1")

        # Initialize graders and chain
        self.retrieval_grader = self._create_retrieval_grader()
        self.hallucination_grader = self._create_hallucination_grader()
        self.answer_grader = self._create_answer_grader()
        self.question_rewriter = self._create_question_rewriter()
        self.rag_chain = self._create_rag_chain()

        # Create state graph
        self.workflow = StateGraph(self.GraphState)
        self.create_graph()

    class GraphState(TypedDict):
        """
        Represents the state of our graph.
        Attributes:
            question: The question to be answered
            thoughts: LLM-generated thoughts
            generation: LLM generation
            documents: List of retrieved documents
        """

        question: str
        thoughts: str
        generation: str
        documents: List[str]
        better_question: str

    def _create_retrieval_grader(self):
        response_schemas = [
            ResponseSchema(name="score", description="a score 'yes' or 'no'", type="string")
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template="""You are a grader assessing the relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} \n
            Please respond in valid JSON format as follows:\n
            {format_instructions}""",
            input_variables=["question", "document"],
            partial_variables={"format_instructions": format_instructions},
        )

        return prompt | self.llm | parser

    def _create_hallucination_grader(self):
        response_schemas = [
            ResponseSchema(name="score", description="a score 'yes' or 'no'", type="string")
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is grounded in a set of facts. \n 
            Here are the facts:
            \n ------- \n
            {documents} 
            \n ------- \n
            Here is the answer: {generation} \n
            Please respond in valid JSON format using the following instructions: \n
            {format_instructions}""",
            input_variables=["generation", "documents"],
            partial_variables={"format_instructions": format_instructions},
        )

        return prompt | self.llm | parser

    def _create_answer_grader(self):
        response_schemas = [
            ResponseSchema(name="score", description="a score 'yes' or 'no'", type="string")
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is useful to resolve a question.
                    Here is the answer:
                    \n ------- \n
                    {generation} 
                    \n ------- \n
                    Here is the question: {question} \n
                    Here is the context: {context}
                    Please respond in valid JSON format using the following instructions: \n
                    {format_instructions}""",
            input_variables=["generation", "question", "context"],
            partial_variables={"format_instructions": format_instructions},
        )

        return prompt | self.llm | parser

    def _create_question_rewriter(self):
        prompt = PromptTemplate(
            template="""You are a question re-writer that paraphrase a question based on the provided context to guide twoard the final answer and. \n
            Here is the initial question: \n\n {question}.
            Here is the context: \n \n {context}
            Just genrete the new question: """,
            input_variables=["question", "context"],
        )
        return prompt | self.llm | StrOutputParser()

    def _create_rag_chain(self):

        response_schemas = [
            ResponseSchema(name="answer", description="the final answer", type="string")
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()
        prompt = PromptTemplate(
            template="""You are a question answering angent that answers a question given the context. \n
            Here is the initial question: \n\n {question}. 
            Here is the context: \n\n {context}. 
            The final answer should directly be respond the question only with no extra information
            Please respond in valid JSON format using the following instructions: \n
            {format_instructions} """,
            input_variables=["question", "context"],partial_variables={"format_instructions": format_instructions},
        )
        return prompt | self.llm | parser

    def generate(self, state):
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        thoughts = state["thoughts"]
        print(thoughts)
        question = f"{thoughts}--{question}"
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        self.last_answer = generation['answer']
        self.iter += 1
        return {"documents": documents, "generation": generation['answer']}

    def transform_query(self, state):
        print("---TRANSFORM QUERY---")
        question = state["question"]
        better_question = self.question_rewriter.invoke({"question": question, "context":state["documents"]})
        self.iter += 1
        return {"documents": state["documents"], "better_question": better_question}

    def grade_generation(self, state):
        print("---CHECK HALLUCINATIONS---")
        score = self.hallucination_grader.invoke(
            {"documents": state["documents"], "generation": state["generation"]}
        )
        if self.iter >= self.max_iter:
            return "useful"
        else:

            if score["score"] == "yes":
                score = self.answer_grader.invoke(
                    {"question": state["question"], "generation": state["generation"], "context": state["documents"]}
                )
                if score["score"] == "yes":
                    return "useful"
                else:
                    return "not useful"
            else:
                return "not supported"

    def get_cot_chain(self):
        # Define the response schema to ensure the JSON format is valid
        response_schemas = [
            ResponseSchema(
                name="thoughts",
                description="The generated reasoning and chain of thought for the question.",
                type="string",
            ),
        ]

        # Initialize a structured output parser based on the response schema
        cot_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = cot_parser.get_format_instructions()

        # Define the prompt template with explicit instructions for valid JSON output
        prompt = """You are tasked with generating a chain of thought for the question: {question}. 
                    You have access to the following context: {context}. 
                    Your job is to provide a detailed chain of reasoning and final thoughts.
                    Please respond with valid JSON using the following instructions: 
                    {format_instructions}"""

        cot_prompt = PromptTemplate(
            template=prompt,
            input_variables=[ "question", "context"],
            partial_variables={"format_instructions": format_instructions},
        )

        # Return the chain of operations: prompt -> llm -> cot_parser
        return cot_prompt | self.llm | cot_parser

    def generate_cot(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE COT---")
        question = state["question"]
        documents = state["documents"]
        cot_chain = self.get_cot_chain()
        # RAG generation
        cot = cot_chain.invoke({"question": question, "context": documents})
        print(cot)
        return {
            "thoughts": cot["thoughts"] + state.get("thoughts", ""),
            "question": question,
        }

    def gene_result(self, state):
        return state

    def create_graph(self):
        # Initialize the workflow
        self.workflow.add_node("generate_cot", self.generate_cot)  # grade documents
        # self.workflow.add_node("grade_documents", self.grade_documents)
        self.workflow.add_node("generate", self.generate)
        self.workflow.add_node("transform_query", self.transform_query)
        self.workflow.add_node("result", self.gene_result)

        self.workflow.add_edge(START, "generate_cot")
        self.workflow.add_edge("transform_query", "result")
        self.workflow.add_edge("generate_cot", "generate")
        self.workflow.add_conditional_edges(
            "generate",
            self.grade_generation,
            {
                "not supported": "generate",
                "useful": "result",
                "not useful": "transform_query",
            },
        )
        # Compile and run
        self.app = self.workflow.compile()

    def run_pipeline(self, question, documents):
        self.iter = 0
        start_time = time.time()
        inputs = {"question": question, "documents": documents}
        final_state = self.app.invoke(inputs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.6f} seconds")
        return final_state


max_iter = 5
max_doc_retrived = 5
cot_self_rag = CoTSelfRAG(max_iter=max_iter,max_doc_retrived=max_doc_retrived)