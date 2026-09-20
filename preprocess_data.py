import pandas as pd
import os
import re
from sqlalchemy import create_engine


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_data')


DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "**",
    "database": "**"
}


def count_tokens(text):
    """
    Simple token estimation.
    Chinese characters, English words, numbers, and punctuation marks are counted separately.
    """
    if not text:
        return 0

    chinese_chars = re.findall(r'[\u4e00-\u9fff]', str(text))
    english_words = re.findall(r'[a-zA-Z]+', str(text))
    numbers = re.findall(r'[0-9]+', str(text))
    punctuations = re.findall(r'[^\w\s\u4e00-\u9fff]', str(text))

    return len(chinese_chars) + len(english_words) + len(numbers) + len(punctuations)


def get_mysql_engine():
    """
    Create a SQLAlchemy engine for MySQL.
    """
    url = (
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(url)


def load_reviews_from_mysql():
    """
    Load review data directly from MySQL by joining Reviews, Products,
    Customers, Categories, and Suppliers.
    """
    engine = get_mysql_engine()

    sql = """
        SELECT
            r.ReviewID,
            r.ProductID,
            r.CustomerID,
            r.Rating,
            r.ReviewText,
            r.ReviewDate,

            p.ProductName,
            p.UnitPrice,
            p.UnitsInStock,

            c.CompanyName,
            c.City,
            c.Country,

            cat.CategoryName,

            s.CompanyName AS SupplierName

        FROM Reviews r
        LEFT JOIN Products p
            ON r.ProductID = p.ProductID
        LEFT JOIN Customers c
            ON r.CustomerID = c.CustomerID
        LEFT JOIN Categories cat
            ON p.CategoryID = cat.CategoryID
        LEFT JOIN Suppliers s
            ON p.SupplierID = s.SupplierID
        ORDER BY r.ReviewDate DESC, r.ReviewID
        """

    df = pd.read_sql(sql, engine)
    print(f"Successfully loaded {len(df)} review records from MySQL")
    return df


def format_review_text(row):
    """
    Format one review record into a structured text description.
    """
    product_name = row.get('ProductName', 'Unknown Product')
    unit_price = row.get('UnitPrice', 'Unknown Price')
    supplier_name = row.get('SupplierName', '')

    customer_id = row.get('CustomerID', 'Unknown Customer ID')
    customer_company = row.get('CompanyName', 'Unknown Customer Company')

    customer_location = f"{row.get('City', '')}, {row.get('Country', '')}"
    customer_location = customer_location.strip(', ')

    category_name = row.get('CategoryName', 'Unknown Category')
    rating = row.get('Rating', 0)
    review_text = row.get('ReviewText', '')

    review_date = row.get('ReviewDate', '')
    try:
        if review_date:
            review_date = pd.to_datetime(review_date).strftime('%Y-%m-%d')
    except Exception:
        pass

    parts = []
    parts.append(f"Customer ID: {customer_id}")
    parts.append(f"Customer Company: {customer_company}")

    if customer_location:
        parts.append(f"Customer Location: {customer_location}")

    parts.append(f"Product Info: {product_name} (Category: {category_name})")

    if supplier_name:
        parts.append(f"Manufacturer: {supplier_name}")

    if unit_price != 'Unknown Price':
        parts.append(f"Product Price: {unit_price}")

    parts.append(f"Rating: {rating} stars")
    parts.append(f"Review Date: {review_date}")
    parts.append(f"Review Content: \"{review_text}\"")

    return "\n".join(parts)


def preprocess_reviews_from_mysql(output_file=None):
    """
    Preprocess review data directly from MySQL and generate processed_reviews.csv.
    """
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, 'processed_reviews.csv')

    df = load_reviews_from_mysql()

    if df.empty:
        print("No review records found in MySQL.")
        return None

    df['text'] = df.apply(lambda row: format_review_text(row), axis=1)
    df['token_count'] = df['text'].apply(count_tokens)

    print(f"Average token count per review: {df['token_count'].mean():.2f}")
    print(f"Maximum token count: {df['token_count'].max()}")
    print(f"Minimum token count: {df['token_count'].min()}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    output_columns = [
        'ReviewID',
        'ProductID',
        'CustomerID',
        'Rating',
        'ReviewDate',
        'text',
        'token_count',
        'CategoryName',
        'ProductName',
        'CompanyName'
    ]

    df[output_columns].to_csv(output_file, index=False, encoding='utf-8')

    print(f"Processed review data has been saved to: {output_file}")
    return output_file


def merge_csv_rows(
    input_file=None,
    output_file=None,
    group_size=5,
    separator="<ROW_SEP>\n\n",
    max_tokens=1000
):
    """
    Merge rows in processed_reviews.csv into larger text chunks.
    Rows are grouped by CategoryName and merged under the max_tokens limit.
    """
    if input_file is None:
        input_file = os.path.join(OUTPUT_DIR, 'processed_reviews.csv')
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, 'merged_reviews.csv')

    print(f"Merging review data with separator: '{separator}'")
    print(f"Maximum token count: {max_tokens}, maximum rows per group: {group_size}")

    if not os.path.exists(input_file):
        print(f"Error: Input file does not exist: {input_file}")
        return None

    df = pd.read_csv(input_file)

    if 'text' not in df.columns:
        print(f"Error: Input file does not contain a text column: {input_file}")
        return None

    all_columns = df.columns.tolist()
    other_columns = [col for col in all_columns if col != 'text']

    merged_rows = []

    if 'token_count' not in df.columns:
        df['token_count'] = df['text'].apply(count_tokens)

    max_single_token = df['token_count'].max()
    if max_tokens < max_single_token:
        print(
            f"Warning: max_tokens ({max_tokens}) is smaller than the maximum "
            f"single-review token count ({max_single_token})"
        )
        max_tokens = max(max_single_token, 500)
        print(f"max_tokens has been adjusted to {max_tokens}")

    if 'CategoryName' in df.columns:
        grouped = df.groupby('CategoryName')

        for category, group in grouped:
            print(f"Processing category: {category}, total reviews: {len(group)}")

            current_texts = []
            current_token_count = 0
            first_row_in_batch = None

            for _, row in group.iterrows():
                if not current_texts:
                    first_row_in_batch = row

                text = row['text']
                token_count = row.get('token_count', count_tokens(text))

                if token_count > max_tokens:
                    print(
                        f"Warning: A single review token count ({token_count}) "
                        f"exceeds the maximum limit ({max_tokens})"
                    )

                    if current_texts:
                        merged_text = separator.join(current_texts)
                        merged_row = {'text': merged_text}

                        for col in other_columns:
                            merged_row[col] = first_row_in_batch[col]

                        merged_row['token_count'] = current_token_count
                        merged_rows.append(merged_row)

                        current_texts = []
                        current_token_count = 0

                    truncated_text = truncate_text(text, max_tokens)
                    truncated_token_count = count_tokens(truncated_text)

                    merged_row = {'text': truncated_text}

                    for col in other_columns:
                        merged_row[col] = row[col]

                    merged_row['token_count'] = truncated_token_count
                    merged_rows.append(merged_row)

                    continue

                separator_tokens = count_tokens(separator) if current_texts else 0

                if current_token_count + token_count + separator_tokens > max_tokens:
                    if current_texts:
                        merged_text = separator.join(current_texts)
                        merged_row = {'text': merged_text}

                        for col in other_columns:
                            merged_row[col] = first_row_in_batch[col]

                        merged_row['token_count'] = current_token_count
                        merged_rows.append(merged_row)

                        current_texts = [text]
                        current_token_count = token_count
                        first_row_in_batch = row
                    else:
                        print("Warning: Unexpected empty batch during token count check")
                else:
                    current_texts.append(text)
                    current_token_count += token_count + separator_tokens

            if current_texts:
                merged_text = separator.join(current_texts)
                merged_row = {'text': merged_text}

                for col in other_columns:
                    merged_row[col] = first_row_in_batch[col]

                merged_row['token_count'] = current_token_count
                merged_rows.append(merged_row)

    else:
        print("CategoryName column not found. Rows will be processed directly.")

        current_texts = []
        current_token_count = 0
        first_row_in_batch = None

        for _, row in df.iterrows():
            if not current_texts:
                first_row_in_batch = row

            text = row['text']
            token_count = row.get('token_count', count_tokens(text))

            separator_tokens = count_tokens(separator) if current_texts else 0

            if current_token_count + token_count + separator_tokens > max_tokens:
                if current_texts:
                    merged_text = separator.join(current_texts)
                    merged_row = {'text': merged_text}

                    for col in other_columns:
                        merged_row[col] = first_row_in_batch[col]

                    merged_row['token_count'] = current_token_count
                    merged_rows.append(merged_row)

                    current_texts = [text]
                    current_token_count = token_count
                    first_row_in_batch = row
            else:
                current_texts.append(text)
                current_token_count += token_count + separator_tokens

                if len(current_texts) >= group_size:
                    merged_text = separator.join(current_texts)
                    merged_row = {'text': merged_text}

                    for col in other_columns:
                        merged_row[col] = first_row_in_batch[col]

                    merged_row['token_count'] = current_token_count
                    merged_rows.append(merged_row)

                    current_texts = []
                    current_token_count = 0
                    first_row_in_batch = None

        if current_texts:
            merged_text = separator.join(current_texts)
            merged_row = {'text': merged_text}

            for col in other_columns:
                merged_row[col] = first_row_in_batch[col]

            merged_row['token_count'] = current_token_count
            merged_rows.append(merged_row)

    merged_df = pd.DataFrame(merged_rows)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # merged_df.insert(0, 'id', range(1, len(merged_df) + 1))
    import uuid

    merged_df.insert(0,"id",[str(uuid.uuid4()) for _ in range(len(merged_df))]
    )

    final_columns = ['id', 'text', 'token_count']

    if 'CategoryName' in merged_df.columns:
        final_columns.append('CategoryName')

    merged_df[final_columns].to_csv(output_file, index=False, encoding='utf-8')

    print(f"Processing completed. Saved to: {output_file}")
    print(f"Original row count: {len(df)}, merged row count: {len(merged_df)}")
    print(f"Final columns: {final_columns}")

    if 'token_count' in merged_df.columns:
        print(f"Average token count after merging: {merged_df['token_count'].mean():.2f}")
        print(f"Maximum token count after merging: {merged_df['token_count'].max()}")
        print(f"Minimum token count after merging: {merged_df['token_count'].min()}")

    if not merged_df.empty:
        first_text = merged_df.iloc[0]['text']
        rows = first_text.split('<ROW_SEP>')
        print(f"\nExample: The first merged record contains {len(rows)} original reviews")
        print("You can use text.split('<ROW_SEP>') to split it back into individual reviews")

    return output_file


def truncate_text(text, max_tokens):
    """
    Truncate text so that its token count does not exceed max_tokens.
    """
    if not text:
        return text

    if count_tokens(text) <= max_tokens:
        return text
#text = text[:1000]
    lines = text.split('\n')
    essential_info = lines[:5]

    review_content_line = None
    for i, line in enumerate(lines):
        if line.startswith("Review Content:"):
            review_content_line = i
            break

    if review_content_line is not None:
        review_content = lines[review_content_line]
        content_match = re.search(r'"(.*)"', review_content)

        if content_match:
            content = content_match.group(1)

            info_tokens = count_tokens('\n'.join(essential_info))
            available_tokens = max_tokens - info_tokens - 20

            if available_tokens <= 0:
                truncated_content = "..."
            else:
                truncated_content = truncate_content(content, available_tokens)

            lines[review_content_line] = f'Review Content: "{truncated_content}..."'

            return '\n'.join(essential_info + [lines[review_content_line]])

    result = '\n'.join(essential_info)

    if count_tokens(result) > max_tokens:
        result = '\n'.join(lines[:3]) + "\n...(content truncated)..."

    return result


def truncate_content(content, max_tokens):
    """
    Truncate review content so that its token count does not exceed max_tokens.
    """
    if not content:
        return content

    if count_tokens(content) <= max_tokens:
        return content

    if re.search(r'[\u4e00-\u9fff]', content):
        left, right = 0, len(content)

        while left < right:
            mid = (left + right) // 2
            if count_tokens(content[:mid]) <= max_tokens:
                left = mid + 1
            else:
                right = mid

        last_sentence_end = max(
            content[:left - 1].rfind('。'),
            content[:left - 1].rfind('！'),
            content[:left - 1].rfind('？'),
            content[:left - 1].rfind('；'),
            content[:left - 1].rfind(','),
            content[:left - 1].rfind('，')
        )

        if last_sentence_end > 0:
            return content[:last_sentence_end + 1]

        return content[:left - 1]

    words = content.split()
    result = []
    current_tokens = 0

    for word in words:
        word_tokens = count_tokens(word)

        if current_tokens + word_tokens <= max_tokens:
            result.append(word)
            current_tokens += word_tokens
        else:
            break

    return ' '.join(result)


if __name__ == "__main__":
    processed_file = preprocess_reviews_from_mysql()

    if processed_file:
        merge_csv_rows(
            input_file=processed_file,
            max_tokens=1000
        )