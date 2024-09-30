from crewai import Agent, Task, Crew
from langchain.llms import Ollama
import os
from crewai_tools import ScrapeWebsiteTool
from get_links import top_matching_links
from crewai_tools import SeleniumScrapingTool

class AI_Agents:
    def __init__(self, support_link_path ,
                 customer_name, email_id,
                 email_subject, email_body):
        os.environ["OPENAI_API_KEY"] = "NA"

        # Running llama3 locally 
        self.llm = Ollama(
            model = "llama3",
            base_url = "http://localhost:11434")
        
        self.support_links_path = support_link_path
        
        # getting required inputs from the email
        self.inputs = {'customer_name': customer_name,
                       'customer_id': email_id,
                       'email_subject': email_subject,
                       'email_body': email_body
                       }
        ####### Creating Agents #########
        self.support_agent = Agent(
            role="Customer Support Representative",
            goal="Deliver exceptional service by resolving user issues efficiently,"
            "enhancing user experience, and supporting the company’s overall objectives.",
            backstory=(
                "You are a customer support executive at coursera (https://www.coursera.org/)"
                "who receives query emails from coursera users and answer them appropriately"
                "if you are not able to answer then politely apologise and direct them to "
                "senior support. Use only the official source to answer "),
            allow_delegation=False,
            max_iter = 10,
            verbose=False,
            llm = self.llm )
        
        self.support_quality_assurance_agent = Agent(
            role="Support Quality Assurance Specialist",
            goal="Get recognition for providing the "
            "best support quality assurance in your team",
            backstory=(
                "You work at coursera (https://www.coursera.org/)and "
                "are now working with your team "
                "on a request from ensuring that "
                "the support representative is "
                "providing the best support possible.\n"
                "You need to make sure that the support representative "
                "is providing full"
                "complete answers, and make no assumptions."),
            allow_delegation=False,
            verbose=False,
            max_iter=10,
            llm = self.llm)
        
        ####### Creating Task #########
        self.inquiry_resolution = Task(
            description=(
                "{customer_name} just reached out with an important query:\n"
                "{email_subject} - {email_body}"
                "Make sure to use everything you know "
                "to provide the best support possible."
                "You must strive to provide a complete "
                "and accurate response to the customer's inquiry."
            ),
            expected_output=(
                "A detailed, informative response in the for of email to the "
                "customer's inquiry that addresses "
                "all aspects of their question.\n"
                "The response should include references "
                "to everything you used to find the answer, "
                "including external data or solutions. "
                "Ensure the answer is complete, "
                "leaving no questions unanswered, and maintain a helpful and friendly "
                "tone throughout."
            ),
            tools=self.matching_links(),
            agent=self.support_agent,
            llm = self.llm)
        
        self.quality_assurance_review = Task(
            description=(
                "Review the response drafted by the Senior Support Representative for {customer_name}'s inquiry. "
                "Ensure that the answer is as comprehensive, accurate, and adheres to the "
                "high-quality standards expected for customer support.\n"
                "Verify that all parts of the customer's inquiry "
                "mentioned in {email_body} have been addressed "
                "thoroughly, with a helpful and friendly tone.\n"
                "Check for references and sources used to "
                "find the information, "
                "ensuring the response is well-supported and "
                "leaves no questions unanswered."
                "If there are any URLs in the response then make sure"
                "they are valid and working if not then remove them"
            ),
            expected_output=(
                "A final, detailed, concise and informative email response "
                "ready to be sent to the customer.\n"
                "This response should fully address the "
                "customer's inquiry, incorporating all "
                "relevant feedback and improvements.\n"
            ),
            agent=self.support_quality_assurance_agent,
            llm = self.llm )
    
    def matching_links(self,):
        ######## Getting matching support links ###############
        ######## Top 3 matching URL with the email subject #####
        get_links = top_matching_links(self.support_links_path)
        link =  get_links.get(self.inputs['email_subject'])

        st_1 = SeleniumScrapingTool(
            website_url=link[0], wait_time=20)
        st_2 = SeleniumScrapingTool(
            website_url=link[1], wait_time=20)
        st_3 = SeleniumScrapingTool(
            website_url=link[2], wait_time=20)
        
        return [st_1, st_2, st_3]
    
    def respond(self,):
        ########## Creating Crew ##########
        crew = Crew(
            agents=[self.support_agent, 
                    self.support_quality_assurance_agent],
            tasks=[self.inquiry_resolution, 
                   self.quality_assurance_review],
            verbose=2,
            memory=False,
            manager_llm = self.llm
            )
        
        ########## Putting AI agents to work ##########
        result = crew.kickoff(inputs=self.inputs)
        
        ########## Cleaning output as a email #########
        formatted_body = result.replace('\\n', '\n').replace('\\\'', '\'')

        subject = "Re: " + self.inputs['email_subject']
        email_content = f"Subject: {subject}\n\n{formatted_body}"
        email_content.replace('[Your Name]', '')
        
        return email_content
                
