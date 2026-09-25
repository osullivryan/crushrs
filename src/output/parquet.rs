//! Minimal columnar Parquet writer (no Arrow dependency): a table is a list
//! of named columns of equal length.

use parquet::basic::Compression;
use parquet::column::writer::ColumnWriter;
use parquet::data_type::ByteArray;
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Column {
    F64(Vec<f64>),
    I64(Vec<i64>),
    Str(Vec<String>),
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Column::F64(v) => v.len(),
            Column::I64(v) => v.len(),
            Column::Str(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn type_decl(&self, name: &str) -> String {
        match self {
            Column::F64(_) => format!("required double {};", name),
            Column::I64(_) => format!("required int64 {};", name),
            Column::Str(_) => format!("required binary {} (UTF8);", name),
        }
    }
}

/// Rows per row group.
const ROW_GROUP: usize = 1 << 18;

/// Write `columns` (name, data) as one Parquet file, snappy compressed.
pub fn write_table(path: &Path, table: &str, columns: &[(&str, Column)]) -> Result<(), String> {
    let n = columns.first().map(|c| c.1.len()).unwrap_or(0);
    for (name, c) in columns {
        if c.len() != n {
            return Err(format!("column '{}' has {} rows, expected {}", name, c.len(), n));
        }
    }
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
    }
    let schema = format!("message {} {{ {} }}", table, columns.iter().map(|(name, c)| c.type_decl(name)).collect::<Vec<_>>().join(" "));
    let schema = Arc::new(parse_message_type(&schema).map_err(|e| e.to_string())?);
    let props = Arc::new(WriterProperties::builder().set_compression(Compression::SNAPPY).build());
    let file = std::fs::File::create(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let mut writer = SerializedFileWriter::new(file, schema, props).map_err(|e| e.to_string())?;
    let mut start = 0;
    while start < n || (n == 0 && start == 0) {
        let end = (start + ROW_GROUP).min(n);
        let mut rg = writer.next_row_group().map_err(|e| e.to_string())?;
        for (_, c) in columns {
            let mut col = rg.next_column().map_err(|e| e.to_string())?.ok_or("schema/column mismatch")?;
            match (col.untyped(), c) {
                (ColumnWriter::DoubleColumnWriter(w), Column::F64(v)) => {
                    w.write_batch(&v[start..end], None, None).map_err(|e| e.to_string())?;
                }
                (ColumnWriter::Int64ColumnWriter(w), Column::I64(v)) => {
                    w.write_batch(&v[start..end], None, None).map_err(|e| e.to_string())?;
                }
                (ColumnWriter::ByteArrayColumnWriter(w), Column::Str(v)) => {
                    let bytes: Vec<ByteArray> = v[start..end].iter().map(|s| ByteArray::from(s.as_str())).collect();
                    w.write_batch(&bytes, None, None).map_err(|e| e.to_string())?;
                }
                _ => return Err("column type mismatch".into()),
            }
            col.close().map_err(|e| e.to_string())?;
        }
        rg.close().map_err(|e| e.to_string())?;
        start = end;
        if n == 0 {
            break;
        }
    }
    writer.close().map_err(|e| e.to_string())?;
    Ok(())
}
