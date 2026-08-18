import asyncio
import asyncpg
import ssl

async def main():
    ctx = ssl.create_default_context(
        cafile="/u/rag-chatbot/data/global-bundle.pem"
    )

    conn = await asyncpg.connect(
        host="database-2.c6dsy0ice6gu.us-east-1.rds.amazonaws.com",
        port=5432,
        user="postgres",
        password="Samrudh111***",
        database="postgres",
        ssl=ctx,
        timeout=30,
    )

    print(await conn.fetch("SELECT version()"))
    await conn.close()

asyncio.run(main())