# Delta Sharing Flask Browser

A lightweight Flask web app to **explore Delta Sharing data resources** — including shares, schemas, tables, metadata, and data previews — **using only Python and Pandas (no Spark required).**
Why I created it? Sometimes it's not that user-friendly to explore what is available in delta shares and having a simple app to browse shares is helpful.

---

## Features

- Browse all available **Shares**, **Schemas**, and **Tables**
- View table **metadata** (schema, version info, file stats)
- **Estimate data size** for tables via file aggregation
- Preview table data directly as a **Pandas DataFrame** (first 20 rows)
- Flexible profile management: upload, paste, environment variable, or file
- Works on any system with Python (Mac, Linux, Windows)

---

## Requirements

- **Python 3.9+**
- Delta Sharing profile file (see below)

---

## Setup

```bash
cd DeltaShareFlask
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Providing Your Delta Sharing Profile

Authenticate using a valid **profile file**. You can:
- Place your profile at `./profiles/config.share`, **or**
- Use the web UI at to upload/paste the profile JSON (stored as `./profiles/config.share`).

**Example profile config.share**:

```json
{
  "shareCredentialsVersion": 1,
  "bearerToken": "...",
  "endpoint": "https://.../api/2.0/delta-sharing/metastores/...",
  "expirationTime": "9999-12-30T23:59:59.480Z"
}
```

---

## Running the App

```bash
python app.py
```

If 5000 is taken, use

```bash
PORT=5050 python app.py
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to access the app.

---

## How It Works

This app uses the [Delta Sharing Python library](https://github.com/delta-io/delta-sharing). Key functions:

- `SharingClient.list_shares()`  
  List all shares.
- `SharingClient.list_schemas(share)`  
  List schemas in a share.
- `SharingClient.list_tables(share, schema)`  
  List tables in a schema.
- `SharingClient.list_files(table)`  
  Aggregate file sizes for data size estimation.
- `SharingClient.get_table_schema(table)`  
  Retrieve table columns info.
- `SharingClient.get_table_version(table)` 
  Current version of the table.
- `delta_sharing.load_as_pandas(Table(...))`  
  Load a shared table into a Pandas DataFrame for preview.


---

## Useful Delta Sharing Library Methods Available

You can use these (without Spark):

- **`list_shares()`**: List all shares
- **`list_schemas(share)`**: List schemas within a share
- **`list_tables(share, schema)`**: List tables within a schema
- **`get_table_schema(share, schema, table)`**: Get table schema
- **`get_table_version(share, schema, table)`**: Get table version
- **`list_files(share, schema, table)`**: Get underlying Delta/Parquet files & estimate table size
- **`load_as_pandas(Table(...))`**: Load entire Delta/Parquet table to a Pandas DataFrame
- **`load_cdf_as_pandas(...)`**: (if available) Change Data Feed to Pandas

See [Delta Sharing Python API](https://github.com/delta-io/delta-sharing/tree/main/python) for more.

---

## Additional Notes

- Size estimation uses `list_files_in_table` or similar API to sum file sizes; may show N/A if unsupported.
- Data preview uses `delta_sharing.load_as_pandas` limited to 20 rows for performance.
- You can manage multiple profiles conveniently using the UI.

---

## Production Deployment

- For production, run with WSGI (e.g., `gunicorn`) and place behind a reverse proxy.
- Never commit credentials or secrets to source control.

---

## License

MIT
