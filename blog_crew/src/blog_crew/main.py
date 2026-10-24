#!/usr/bin/env python
import sys
import warnings
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from blog_crew.crew import BlogCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew to write a blog post and generate an analytical report of that blog.
    """
    topic = sys.argv[1] if len(sys.argv) > 1 else 'The Future of Artificial Intelligence in Gaming'
    inputs = {
        'topic': topic
    }

    print(f"\n[Starting Blog Crew for Topic: '{topic}']\n")

    try:
        result = BlogCrew().crew().kickoff(inputs=inputs)
        print("\n\n################################################")
        print("## CREW EXECUTION COMPLETED SUCCESSFULLY!     ##")
        print("################################################\n")
        print("Generated files:")
        print(" - Blog Post:   output/blog_post.md")
        print(" - Blog Report: output/blog_report.md\n")
        print("Final Report Output:\n")
        print(result)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

if __name__ == "__main__":
    run()
