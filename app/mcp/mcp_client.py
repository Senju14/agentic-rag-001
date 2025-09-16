import asyncio
import os
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from dotenv import load_dotenv
load_dotenv()


PUBLIC_URL = os.environ.get("MCP_PUBLIC_URL")
PRIVATE_URL = os.environ.get("MCP_PRIVATE_URL")

async def main():
    # Test with public server
    transport = StreamableHttpTransport(url=PUBLIC_URL)
    client = Client(transport)
    async with client:
        print("[Public] Server reachable!\n", await client.ping())
        print("[Public] Tools:", await client.list_tools())

        print("\n[Public] Available resources:", await client.list_resources())
        print("\n[Public] Available prompts:", await client.list_prompts())

        # Test tool: search topic
        print(await client.call_tool("search_topic", {"query": "Machine learning"}))

        # Test tool: math solver
        print(await client.call_tool("math_solver", {"expression": "2+2*5"}))

        # Test tool: password generator
        print(await client.call_tool("password_generator", {"length": 16, "use_special": True}))

        # Test resource
        print(await client.read_resource("resource://tech/trends"))

        # Test prompt
        print(await client.get_prompt("explore_topic_prompt", {"topic": "AI in healthcare"}))



    # Test with private server
    transport_p = StreamableHttpTransport(url=PRIVATE_URL)
    client_p = Client(transport_p)
    async with client_p:
        print("[Private] Server reachable!\n", await client_p.ping())
        print("[Private] Tools:", await client_p.list_tools())
        # Test search_in_database
        print(await client_p.call_tool("search_in_database", {"query": "law", "top_k": 3}))
        # Test send_mail
        print(await client_p.call_tool("send_mail", {"to_email": "nng.ai.intern01@gmail.com", "subject": "Test MCP", "body": "Hello from MCP private!"}))

# python -m app.mcp.mcp_client

if __name__ == "__main__":
    asyncio.run(main())
