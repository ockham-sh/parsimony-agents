import asyncio
import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

from ockham_agents import Agent
from ockham.connectors.fred import CONNECTORS as FRED

load_dotenv()

async def run_pattern(name: str, agent: Agent, prompt: str, ctx=None):
    print(f"\n{'='*50}")
    print(f"Pattern: {name}")
    print(f"Prompt: '{prompt}'")
    print(f"{'='*50}")
    
    result = await agent.ask(prompt, ctx=ctx)
    
    print("\n[Agent Response]")
    print(result.text)
    
    print("\n[Returned Datasets]")
    for var_name, dataset in result.datasets.items():
        print(f" - {var_name}: {dataset.title} (notebook: {dataset.notebook_refs})")
        if dataset.variable_preview:
            # Print columns if available in the preview
            cols = dataset.variable_preview.get("columns", [])
            print(f"   Columns: {[c.get('name') for c in cols]}")

    print("\n[Returned Charts]")
    for var_name, chart in result.charts.items():
        print(f" - {var_name}: {chart.title} (source dataset: {chart.source_dataset_variable_name})")
        
    print("\n[Generated Code]")
    for nb_name, notebook in result.code.items():
        print(f"--- Notebook: {nb_name} ---")
        print(notebook.code.strip())
        
    return result

async def main():
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        print("Error: FRED_API_KEY not found in environment.")
        return
        
    agent = Agent(
        model="gemini/gemini-3-flash-preview", # Using gemini as it is available
        connectors=FRED.bind_deps(api_key=fred_key)
    )

    # Pattern 1: Single dataset retrieval
    res1 = await run_pattern(
        "1. Single Dataset Retrieval", 
        agent, 
        "Fetch the US Unemployment Rate (UNRATE) from FRED and return it as a dataset."
    )
    
    # Pattern 2: Data Transformation and Charting (using previous context)
    res2 = await run_pattern(
        "2. Transformation and Charting",
        agent,
        "Calculate the 12-month moving average of the unemployment rate and plot it as a line chart.",
        ctx=res1.context
    )
    
    # Pattern 3: Complex Multi-Series Retrieval and Comparison
    agent2 = Agent(
        model="gemini/gemini-3-flash-preview",
        connectors=FRED.bind_deps(api_key=fred_key)
    )
    
    await run_pattern(
        "3. Multi-Series Comparison",
        agent2,
        "Fetch both the US GDP (GDPC1) and the Consumer Price Index (CPIAUCSL). Merge them into a single dataset, calculate the year-over-year percentage change for both, and return the combined dataset and a chart comparing their growth rates since 2000."
    )

if __name__ == "__main__":
    asyncio.run(main())
