---
title: "cibica - Data Schema"
author: "Torbjörn E. M. Nordling"
date: "2026-06-16"
license: "Apache-2.0"
version: 1.0.0
purpose: "Schema documentation for dataset columns and types."
---

# Data Schema

## Dataset: cibica

## Columns

| Column Name | Data Type | Description | Example | Missing Values |
|-------------|-----------|-------------|---------|----------------|
| id | integer | Unique identifier | 12345 | No |
| name | string | [Description] | "Example" | No |
| value | float | [Description] | 123.45 | Yes (0.5%) |
| category | string | [Description] | "A" | No |
| timestamp | datetime | [Description] | "2024-01-01 12:00:00" | No |

## Data Types

- **integer**: Whole numbers
- **float**: Decimal numbers
- **string**: Text data
- **datetime**: Date and time in ISO 8601 format
- **boolean**: True/False values

## Categorical Variables

### category
- **A**: [Description]
- **B**: [Description]
- **C**: [Description]

## Units

- **value**: [units, e.g., meters, dollars, etc.]

## Constraints

- All values in `value` column must be non-negative
- `timestamp` must be within range [start_date, end_date]
- `category` must be one of {A, B, C}

## File Format

- **Format**: CSV
- **Encoding**: UTF-8
- **Delimiter**: Comma (,)
- **Header**: Yes (first row)
- **Null Values**: Represented as empty string or "NA"
