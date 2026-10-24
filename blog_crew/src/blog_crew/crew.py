from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class BlogCrew:
    """BlogCrew crew for writing a blog post and generating an analytical report of that blog."""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config['writer'],
            verbose=True
        )

    @agent
    def reporter(self) -> Agent:
        return Agent(
            config=self.agents_config['reporter'],
            verbose=True
        )

    @task
    def blog_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['blog_writing_task'],
            output_file='output/blog_post.md'
        )

    @task
    def blog_reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['blog_reporting_task'],
            context=[self.blog_writing_task()],
            output_file='output/blog_report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the BlogCrew crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
