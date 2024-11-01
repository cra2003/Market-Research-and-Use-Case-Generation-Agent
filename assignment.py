import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import AzureChatOpenAI
from crewai_tools import SerperDevTool, \
                         ScrapeWebsiteTool, \
                         WebsiteSearchTool
from dotenv import load_dotenv


load_dotenv() 


serper_tool = SerperDevTool()


os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY') 
llm_azure=AzureChatOpenAI(
    openai_api_version="2024-02-01",
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_API_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)
def run_crew(topic):
    industry_researcher = Agent(
        role="Industry Researcher",
        goal="Research on a company or industry called {topic} to understand which domain this company or industry falls into (e.g., Automotive, Manufacturing, Finance, Retail, Healthcare, etc.), key offerings, and strategic focus.",
        backstory="You are tasked with researching the industry where the specified company operates. "
                "Do a proper competitor analysis"
                "Segment the company into its respective domain."
                "Gather information about the products or services offered by the company or industry"
                "Gather information about industry trends, key players, and the company's strategic focus areas.",
        allow_delegation=False,
        verbose=True,
        tools=[serper_tool,ScrapeWebsiteTool(),WebsiteSearchTool()]
        # Add the search tool to this agent
    )

    use_case_generator = Agent(
        role="Use Case Generator",
        goal="Analyze industry standards and propose relevant AI and ML use cases which are feasible and practical for the company and where AI can be used in the industry or company's products", 
        backstory="You will analyze market trends and standards in the specified industry. "
                "Refer to reports and insights on AI and digital transformation from industry-specific sources"
                "Search for industry-specific use cases (e.g., “how is the retail industry leveraging AI and ML” or “AI applications in automotive manufacturing”)."
                "Based on your analysis, propose actionable use cases leveraging AI and Generative AI technologies.",
        allow_delegation=False,
        verbose=True
    )

    resource_collector = Agent(
        role="Resource Collector",
        goal="Collect relevant datasets and resources for the proposed use cases.",
        backstory="You are responsible for gathering relevant datasets from platforms like Kaggle, HuggingFace and GitHub. "
                "Also, ensure to save the resource links for easy access.",
        allow_delegation=False,
        tools=[serper_tool,ScrapeWebsiteTool(),WebsiteSearchTool()],
        verbose=True
    )

    finalizer = Agent(
        role="Finalizer",
        goal="Compile a cohesive, well-structured proposal document, integrating research findings, top use cases, and relevant resources.",
        backstory=(
            "As the Proposal Finalizer, you are responsible for creating a polished, insightful document that highlights key use cases relevant to the company's goals. "
            "Provide references for each use case to indicate where the inspiration came from. "
            "Organize the proposal clearly, ensuring all resource links are clickable and that references are cited appropriately."
        ),
        allow_delegation=False,
        verbose=True
    )



    industry_research_task = Task(
        description=(
            "1. Conduct market research on company called {topic} using available tools.\n"
            "2. A vision and product information on the industry \n"
            "3. Identify the company's key offerings and strategic focus areas(e.g., operations, supply chain, customer experience, etc.)\n"
            "4. Gather insights on industry trends and competitor analysis."
        ),
        expected_output="A comprehensive report on the industry and company, detailing key offerings, trends, and competitive landscape.",
        agent=industry_researcher,
        output_file="industry_research.md"
    )

    use_case_generation_task = Task(
        description=(
            "1. Analyze the collected industry data and standards.\n"
            "2. Propose relevant use cases leveraging AI and Generative AI technologies.\n"
            "3. Ensure the use cases improve operational efficiency and customer satisfaction."
            "4. Search for industry-specific use cases (e.g., “how is the retail industry leveraging AI and ML” or “AI applications in automotive manufacturing”)."
        ),
        expected_output="A list of relevant AI and Generative AI use cases tailored for the company.",
        agent=use_case_generator
    )

    resource_collection_task = Task(
        description=(
            "1. Search for relevant datasets on Kaggle, HuggingFace, and GitHub.\n"
            "2. Compile links and resources related to the proposed use cases.\n"
            "3. Save the resource links in a text or markdown file."
        ),
        expected_output="A text or markdown file containing links to relevant datasets and resources for the proposed use cases.",
        agent=resource_collector,
        output_file="datasets_links.md"
    )

    proposal_task = Task(
        description=(
            "1. Compile the research, identified use cases, and resource links into a professional final proposal document.\n"
            "2. List the top use cases relevant to the company\'s goals and operational needs that are practical and feasible\n"
            "3. Include references for each suggested use case, specifying the source that inspired it.\n"
            "4. Ensure all resource links are clickable and properly cited in the document."
        ),
        expected_output="A well-organized proposal document with a clear list of prioritized use cases, complete with references and accessible resource links.",
        agent=finalizer,
        output_file="use_case_report.md"
    )



    crew = Crew(
        agents=[industry_researcher,use_case_generator,resource_collector,finalizer],
        tasks=[industry_research_task,use_case_generation_task,resource_collection_task,proposal_task],
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff(inputs={"topic": topic})


