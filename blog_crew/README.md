# blog_crew

A CrewAI project that writes a blog post on a given topic and generates a comprehensive analytical report of that blog.

## Running

You can run the crew directly with Python:

```bash
python main.py
```

Or pass a custom topic:

```bash
python main.py "Advances in Quantum Computing"
```

Or using the CrewAI CLI:

```bash
crewai run
```

## Generated Outputs

When execution completes, two files are saved in the `output/` directory:
- `output/blog_post.md`: The complete markdown blog post written by the **writer** agent.
- `output/blog_report.md`: The analytical report evaluating the blog post written by the **reporter** agent.

## Project Structure

- `main.py` - Root runner script
- `src/blog_crew/`
  - `crew.py` - Crew definition with `@CrewBase`, `@agent`, and `@task`
  - `main.py` - Entrypoint function (`run()`)
  - `config/agents.yaml` - Agent configurations (`writer`, `reporter`)
  - `config/tasks.yaml` - Task configurations (`blog_writing_task`, `blog_reporting_task`)
  - `tools/` - Custom tools
- `.env` - Environment variables and API keys
