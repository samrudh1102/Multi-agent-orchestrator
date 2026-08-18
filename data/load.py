# cat << 'EOF' > load.py
import os, glob, psycopg2, boto3

# Auto-generate temporary AWS IAM auth token
rds_client = boto3.client('rds', region_name='us-east-1')
auth_token = rds_client.generate_db_auth_token(
    DBHostname=os.environ['PGHOST'],
    Port=int(os.environ['PGPORT']),
    DBUsername=os.environ['PGUSER'],
    Region='us-east-1'
)

print("==> Generated AWS IAM Database Token successfully")

conn = psycopg2.connect(
    host=os.environ["PGHOST"],
    port=os.environ["PGPORT"],
    user=os.environ["PGUSER"],
    password=auth_token,
    dbname=os.environ["PGDATABASE"],
    sslmode="require"
)
cur = conn.cursor()

print("==> Applying schema files")
for file in sorted(glob.glob("./out_postgres/schema/*.sql")):
    print(f"    Applying {file}")
    cur.execute(open(file).read())

print("==> Loading seed.sql data")
cur.execute(open("./out_postgres/seed.sql").read())

conn.commit()

cur.execute("SELECT count(*) FROM test_scenarios;")
scenarios = cur.fetchone()[0]
cur.execute("SELECT count(*) FROM rides;")
rides = cur.fetchone()[0]

print(f"\n==> SUCCESS! Scenarios loaded: {scenarios}, Rides loaded: {rides}")
cur.close()
conn.close()
# EOF