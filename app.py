import os
import json
import re
from typing import Optional, Tuple, Any

from flask import Flask, render_template, redirect, url_for, request, flash, send_from_directory, Response
import delta_sharing
import pandas as pd

# Flask setup
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")

# Profile handling
DEFAULT_PROFILE_PATH = os.environ.get("DELTA_SHARING_PROFILE", os.path.abspath("config.share"))
PROFILES_DIR = os.path.abspath("profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)


def get_profile_path() -> Optional[str]:
	# Priority: querystring -> env/default -> profiles dir fallback
	path = request.args.get("profile") or DEFAULT_PROFILE_PATH
	if path and os.path.exists(path):
		return path
	candidate = os.path.join(PROFILES_DIR, "config.share")
	if os.path.exists(candidate):
		return candidate
	return None


def get_client() -> Tuple[Optional[delta_sharing.SharingClient], Optional[str]]:
	profile_path = get_profile_path()
	if not profile_path:
		return None, None
	try:
		client = delta_sharing.SharingClient(profile_path)
		return client, profile_path
	except Exception as exc:
		flash(f"Failed to initialize Delta Sharing client: {exc}", "error")
		return None, profile_path


def build_sidebar_lists(client: delta_sharing.SharingClient, current_share: Optional[str], current_schema: Optional[str]):
	"""Return lists for sidebar tree: all shares, schemas of current share, tables of current schema."""
	shares_all = []
	schemas_in_share = []
	tables_in_schema = []
	try:
		shares_all = client.list_shares()
	except Exception:
		shares_all = []
	if current_share:
		try:
			schemas_in_share = client.list_schemas(delta_sharing.Share(current_share))
		except Exception:
			schemas_in_share = []
	if current_share and current_schema:
		try:
			tables_in_schema = client.list_tables(delta_sharing.Schema(current_schema, current_share))
		except Exception:
			tables_in_schema = []
	return shares_all, schemas_in_share, tables_in_schema


@app.route("/")
def index():
	client, profile_path = get_client()
	if not client:
		return render_template("index.html", has_profile=False, profile_path=profile_path)
	return redirect(url_for("list_shares"))


@app.route("/profile", methods=["GET", "POST"])
def profile():
	if request.method == "POST":
		upload = request.files.get("file")
		text = request.form.get("json_text", "").strip()
		out_path = os.path.join(PROFILES_DIR, "config.share")
		try:
			if upload and upload.filename:
				upload.save(out_path)
				flash(f"Profile uploaded to {out_path}", "success")
			elif text:
				_ = json.loads(text)
				with open(out_path, "w", encoding="utf-8") as f:
					f.write(text)
				flash(f"Profile saved to {out_path}", "success")
			else:
				flash("Provide a file or JSON profile.", "error")
		except Exception as exc:
			flash(f"Failed saving profile: {exc}", "error")
		return redirect(url_for("index"))
	
	# Check if profile already exists
	has_existing_profile = False
	existing_profile_path = None
	if os.path.exists(os.path.join(PROFILES_DIR, "config.share")):
		has_existing_profile = True
		existing_profile_path = os.path.join(PROFILES_DIR, "config.share")
	elif os.path.exists(DEFAULT_PROFILE_PATH):
		has_existing_profile = True
		existing_profile_path = DEFAULT_PROFILE_PATH
	
	return render_template("profile.html", 
		default_path=DEFAULT_PROFILE_PATH,
		has_existing_profile=has_existing_profile,
		existing_profile_path=existing_profile_path)


@app.route("/shares")
def list_shares():
	client, profile_path = get_client()
	if not client:
		return redirect(url_for("index"))
	shares = client.list_shares()
	shares_all, schemas_in_share, tables_in_schema = build_sidebar_lists(client, None, None)
	return render_template(
		"shares.html",
		shares=shares,
		profile_path=profile_path,
		shares_all=shares_all,
		schemas_in_share=schemas_in_share,
		tables_in_schema=tables_in_schema,
	)


@app.route("/shares/<share>/schemas")
def list_schemas(share: str):
	client, profile_path = get_client()
	if not client:
		return redirect(url_for("index"))
	share_obj = delta_sharing.Share(share)
	schemas = client.list_schemas(share_obj)
	shares_all, schemas_in_share, tables_in_schema = build_sidebar_lists(client, share, None)
	return render_template(
		"schemas.html",
		share=share,
		schemas=schemas,
		profile_path=profile_path,
		shares_all=shares_all,
		schemas_in_share=schemas_in_share,
		tables_in_schema=tables_in_schema,
	)


@app.route("/shares/<share>/<schema>/tables")
def list_tables(share: str, schema: str):
	client, profile_path = get_client()
	if not client:
		return redirect(url_for("index"))
	schema_obj = delta_sharing.Schema(schema, share)
	tables = client.list_tables(schema_obj)
	shares_all, schemas_in_share, tables_in_schema = build_sidebar_lists(client, share, schema)
	return render_template(
		"tables.html",
		share=share,
		schema=schema,
		tables=tables,
		profile_path=profile_path,
		shares_all=shares_all,
		schemas_in_share=schemas_in_share,
		tables_in_schema=tables_in_schema,
	)


@app.route("/shares/<share>/<schema>/tables/<table>")
def table_details(share: str, schema: str, table: str):
	client, profile_path = get_client()
	if not client:
		return redirect(url_for("index"))

	# Active tab: overview | files | preview (history removed)
	tab = request.args.get("tab", "overview")

	# Table constructor expects (name, share, schema)
	ds_table = delta_sharing.Table(table, share, schema)

	metadata_json: Optional[str] = None
	metadata_error: Optional[str] = None
	metadata_summary: Optional[dict] = None
	schema_columns: Optional[list] = None
	# Summary size info for right-hand panel
	num_files: Optional[int] = None
	total_size: Optional[int] = None

	# Overview tab: metadata
	if tab == "overview":
		try:
			if hasattr(client, "_rest_client") and hasattr(client._rest_client, "query_table_metadata"):
				raw_meta: Any = client._rest_client.query_table_metadata(ds_table)
				# Prefer attribute-based unpacking when response is an object
				if hasattr(raw_meta, "metadata"):
					proto = getattr(raw_meta, "protocol", None)
					meta = getattr(raw_meta, "metadata", None)
					parsed_schema: Any = None
					if meta is not None:
						schema_str = getattr(meta, "schema_string", None)
						if isinstance(schema_str, (str, bytes)):
							try:
								parsed_schema = json.loads(schema_str)
							except Exception:
								parsed_schema = schema_str.decode("utf-8") if isinstance(schema_str, bytes) else schema_str
					format_obj = getattr(meta, "format", None)
					meta_dict = {
						"id": getattr(meta, "id", None),
						"name": getattr(meta, "name", None),
						"description": getattr(meta, "description", None),
						"format": {
							"provider": getattr(format_obj, "provider", None) if format_obj is not None else None,
							"options": getattr(format_obj, "options", {}) if format_obj is not None else {},
						},
						"schema": parsed_schema,
						"configuration": getattr(meta, "configuration", {}),
						"partition_columns": getattr(meta, "partition_columns", []),
						"version": getattr(meta, "version", None),
						"size": getattr(meta, "size", None),
						"num_files": getattr(meta, "num_files", None),
						"created_time": getattr(meta, "created_time", None),
					}
					meta_obj = {
						"delta_table_version": getattr(raw_meta, "delta_table_version", None),
						"protocol": {
							"min_reader_version": getattr(proto, "min_reader_version", None) if proto is not None else None,
							"min_writer_version": getattr(proto, "min_writer_version", None) if proto is not None else None,
							"reader_features": getattr(proto, "reader_features", None) if proto is not None else None,
							"writer_features": getattr(proto, "writer_features", None) if proto is not None else None,
						},
						"metadata": meta_dict,
					}
				elif isinstance(raw_meta, (str, bytes)):
					try:
						meta_obj = json.loads(raw_meta)
					except Exception:
						meta_obj = {"raw": raw_meta.decode("utf-8") if isinstance(raw_meta, bytes) else raw_meta}
				elif isinstance(raw_meta, dict):
					meta_obj = raw_meta
				else:
					meta_obj = json.loads(json.dumps(raw_meta, default=str))
				metadata_json = json.dumps(meta_obj, indent=2)
				try:
					m = meta_obj.get("metadata", {}) if isinstance(meta_obj, dict) else {}
					p = meta_obj.get("protocol", {}) if isinstance(meta_obj, dict) else {}
					schema_obj = m.get("schema") if isinstance(m, dict) else None
					fields = schema_obj.get("fields", []) if isinstance(schema_obj, dict) else []
					schema_columns = [
						{"name": f.get("name"), "type": f.get("type"), "nullable": f.get("nullable")}
						for f in fields if isinstance(f, dict)
					]
					metadata_summary = {
						"delta_table_version": meta_obj.get("delta_table_version"),
						"format_provider": (m.get("format") or {}).get("provider") if isinstance(m.get("format"), dict) else None,
						"num_files": m.get("num_files"),
						"size": m.get("size"),
						"version": m.get("version"),
						"partition_columns_count": len(m.get("partition_columns") or []),
						"min_reader_version": p.get("min_reader_version"),
						"min_writer_version": p.get("min_writer_version"),
					}
					# Populate summary values for display if available
					num_files = metadata_summary.get("num_files")
					total_size = metadata_summary.get("size")
				except Exception:
					metadata_summary = None
		except Exception as exc:
			resp = getattr(exc, "response", None)
			if resp is not None:
				try:
					body = resp.json()
					details = body.get("details", []) if isinstance(body, dict) else []
					is_dv = False
					for item in details:
						meta = (item or {}).get("metadata", {})
						if isinstance(meta, dict) and meta.get("dsError") == "DS_UNSUPPORTED_DELTA_TABLE_FEATURES" and meta.get("tableFeatures") == "delta.enableDeletionVectors":
							is_dv = True
							break
				except Exception:
					is_dv = False
				if is_dv:
					alter_stmt = f"ALTER TABLE {schema}.{table} SET TBLPROPERTIES (delta.enableDeletionVectors=false)"
					metadata_error = (
						"This table has Deletion Vectors enabled. Metadata via REST requires a Delta response. "
						"Options: disable DVs with <strong><code>" + alter_stmt + "</code></strong>, "
						"or query with a DV-capable runtime (DBR &gt;= 14.1 or delta-sharing-spark &gt;= 3.1 with responseFormat=delta)."
					)
				else:
					metadata_error = str(exc)
			else:
				metadata_error = str(exc)

	# Files tab: list files
	files = None
	if tab == "files":
		try:
			if hasattr(client, "list_files_in_table"):
				files = client.list_files_in_table(ds_table)
				# If metadata lacked size info, compute here
				if files is not None and (num_files is None or total_size is None):
					num_files = len(files)
					total_size = sum(getattr(f, "size", 0) for f in files)
		except Exception as exc:
			flash(f"Failed to list files: {exc}", "error")

    # History functionality removed

	# Preview tab: filtering, columns, pagination
	preview_df = None
	preview_error = None
	# Pagination params
	try:
		page = max(1, int(request.args.get("page", 1)))
	except Exception:
		page = 1
	try:
		page_size = int(request.args.get("page_size", 20))
		if page_size <= 0:
			page_size = 20
		if page_size > 500:
			page_size = 500
	except Exception:
		page_size = 20

	# Column selection and simple filter (pandas query)
	selected_columns = request.args.get("columns", "").strip()
	filter_expr = request.args.get("where", "").strip()
	
	# Time travel and comparison removed

	fetch_limit = min(page * page_size, 5000)
	try:
		profile_url = f"{profile_path}#{share}.{schema}.{table}"
		columns_arg = None
		if selected_columns:
			columns_arg = [c.strip() for c in selected_columns.split(",") if c.strip()]
			if not columns_arg:
				columns_arg = None
		# Load data (no time travel)
		full_df = delta_sharing.load_as_pandas(profile_url, limit=fetch_limit)
		if isinstance(full_df, pd.DataFrame):
			# Convert dtypes to proper types for better filtering
			full_df = full_df.convert_dtypes()
			# Also try to convert object columns that look numeric to numeric types
			for col in full_df.columns:
				if full_df[col].dtype == 'object':
					# Try converting to numeric, coercing errors to NaN
					try:
						converted = pd.to_numeric(full_df[col], errors='coerce')
						# If we successfully converted most values, use the converted version
						if converted.notna().sum() > len(full_df) * 0.5:
							full_df[col] = converted
					except Exception:
						pass
			# Apply column selection after load if requested
			if columns_arg:
				existing = [c for c in columns_arg if c in full_df.columns]
				if existing:
					full_df = full_df[existing]
			# Apply client-side filter if provided
			if filter_expr and not full_df.empty:
				try:
					normalized = _normalize_query_expr(filter_expr, list(full_df.columns))
					# Try query first
					try:
						full_df = full_df.query(normalized, engine="python")
					except (TypeError, ValueError) as e:
						# If query fails due to type issues, use boolean indexing with eval
						# This handles type coercion better
						mask = full_df.eval(normalized)
						full_df = full_df[mask]
				except Exception as e:
					# Store filter error for debugging
					preview_error = f"Filter error: {str(e)}"
			start = (page - 1) * page_size
			end = start + page_size
			preview_df = full_df.iloc[start:end]
	except Exception as exc:
		preview_error = str(exc)
	
	# Version comparison removed

	def _url_for_page(p: int) -> str:
		args = request.args.to_dict(flat=True)
		args["page"] = str(p)
		args["page_size"] = str(page_size)
		args["tab"] = tab
		return url_for("table_details", share=share, schema=schema, table=table, **args)

	prev_url = _url_for_page(page - 1) if page > 1 else None
	next_url = _url_for_page(page + 1) if preview_df is not None and len(preview_df) == page_size else None

	shares_all, schemas_in_share, tables_in_schema = build_sidebar_lists(client, share, schema)
	# Human-readable size for UI
	total_size_human = _format_bytes(total_size) if isinstance(total_size, int) else None

	return render_template(
		"table.html",
		share=share,
		schema=schema,
		table=table,
		tab=tab,
		metadata_json=metadata_json,
		metadata_error=metadata_error,
		metadata_summary=metadata_summary,
		schema_columns=schema_columns,
		files=files,
		table_history=None,
		preview_df=preview_df,
		preview_error=preview_error,
		comparison_data=None,
		num_files=num_files,
		total_size=total_size,
		total_size_human=total_size_human,
		selected_columns=selected_columns,
        filter_expr=filter_expr,
        page=page,
        page_size=page_size,
		prev_url=prev_url,
		next_url=next_url,
		profile_path=profile_path,
		shares_all=shares_all,
		schemas_in_share=schemas_in_share,
		tables_in_schema=tables_in_schema,
	)


@app.get('/shares/<share>/<schema>/tables/<table>/export')
def export_table_preview(share: str, schema: str, table: str):
	client, profile_path = get_client()
	if not client:
		return redirect(url_for("index"))
	fmt = request.args.get("format", "csv").lower()
	page_size = int(request.args.get("page_size", 1000))
	if page_size > 10000:
		page_size = 10000
	profile_url = f"{profile_path}#{share}.{schema}.{table}"
	df = delta_sharing.load_as_pandas(profile_url, limit=page_size)
	if not isinstance(df, pd.DataFrame):
		flash("No data to export", "error")
		return redirect(url_for("table_details", share=share, schema=schema, table=table))
	if fmt == "parquet":
		buf = df.to_parquet(index=False)
		return Response(buf, headers={
			"Content-Type": "application/octet-stream",
			"Content-Disposition": f"attachment; filename={table}.parquet"
		})
	# default csv
	csv = df.to_csv(index=False)
	return Response(csv, headers={
		"Content-Type": "text/csv",
		"Content-Disposition": f"attachment; filename={table}.csv"
	})


@app.route('/profiles/<path:filename>')
def serve_profile(filename: str):
	return send_from_directory(PROFILES_DIR, filename)


def _normalize_query_expr(expr: str, columns: list) -> str:
	# Wrap column identifiers in backticks to support spaces/special chars
	if not expr:
		return expr
	result = expr
	# Sort by length to avoid partial replacements
	for col in sorted([str(c) for c in columns], key=len, reverse=True):
		# skip if already quoted/backticked
		if col.startswith("`") and col.endswith("`"):
			continue
		pattern = r"(?<![`\w])" + re.escape(col) + r"(?![`\w])"
		result = re.sub(pattern, f"`{col}`", result)
	return result


def _format_bytes(num_bytes: Optional[int]) -> Optional[str]:
	"""Convert a byte count to a human-readable string (KB, MB, GB, TB)."""
	if num_bytes is None:
		return None
	units = ["bytes", "KB", "MB", "GB", "TB", "PB"]
	value = float(num_bytes)
	unit_index = 0
	while value >= 1024.0 and unit_index < len(units) - 1:
		value /= 1024.0
		unit_index += 1
	if unit_index == 0:
		return f"{int(value)} {units[unit_index]}"
	return f"{value:.2f} {units[unit_index]}"


def _compare_dataframes(*args, **kwargs):
    # Placeholder retained to avoid import issues if referenced elsewhere; not used.
    return {}


if __name__ == "__main__":
	port = int(os.environ.get("PORT", 5000))
	app.run(host="0.0.0.0", port=port, debug=True)
