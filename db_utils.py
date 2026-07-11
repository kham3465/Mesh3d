# db_utils.py
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
import numpy as np
import os

# --- THAY ĐỔI CÁC THÔNG SỐ NÀY CHO PHÙ HỢP VỚI DB CỦA BẠN ---
DB_CONFIG = {
    "dbname": os.environ.get("DB_NAME", "face_recog_db"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", "postgres"), # Mật khẩu này được quản lý bởi Docker Compose
    "host": os.environ.get("DB_HOST", "database-iot.c7oewgusah9l.ap-southeast-1.rds.amazonaws.com"),
    "port": os.environ.get("DB_PORT", "5432")
}
# ---------------------------------------------------------

def get_db_connection():
    """Tạo và trả về một kết nối đến DB."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.OperationalError as e:
        print(f"Lỗi kết nối đến database: {e}")
        print("Vui lòng kiểm tra lại thông tin kết nối trong file db_utils.py và đảm bảo PostgreSQL đang chạy.")
        return None

def setup_database():
    """
    Chạy một lần duy nhất để thiết lập DB: tạo extension, bảng và chỉ mục.
    """
    conn = get_db_connection()
    if conn is None:
        return

    with conn.cursor() as cur:
        print("1. Kích hoạt extension 'vector'...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn)

        print("2. Tạo bảng 'identities' nếu chưa tồn tại...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS identities (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL,
                embedding VECTOR(512), 
                image_url VARCHAR(512),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)

        print("3. Tạo chỉ mục (index) để tăng tốc tìm kiếm...")
        # Tạo chỉ mục IVFFlat để tăng tốc tìm kiếm.
        # `lists` nên bằng sqrt(số lượng bản ghi dự kiến)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_identities_embedding ON identities USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
        
        print("4. Tạo bảng 'recognition_logs' để lưu lịch sử nhận dạng...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id SERIAL PRIMARY KEY,
                identity_name VARCHAR(255) NOT NULL,
                similarity FLOAT NOT NULL,
                recognized_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        
        print("5. Tạo chỉ mục (index) trên bảng logs để truy vấn nhanh hơn...")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON recognition_logs (recognized_at DESC);")

        conn.commit()
    conn.close()
    print("Thiết lập database hoàn tất!")

def enroll_person_db(embedding: np.ndarray, image_url: str = None):
    """Lưu một người dùng mới vào DB và trả về ID của họ."""
    conn = get_db_connection()
    if conn is None:
        return -1, "Lỗi: Không thể kết nối đến database."

    register_vector(conn)
    with conn.cursor() as cur:
        try:
            # Thêm người mới và lấy ID được tạo tự động. Tên ban đầu được gán bằng ID.
            # Bước 1: Chèn bản ghi với một tên tạm thời để lấy ID
            cur.execute(
                """
                INSERT INTO identities (name, embedding, image_url) 
                VALUES (md5(random()::text || clock_timestamp()::text)::uuid::text, %s, %s) 
                RETURNING id;
                """,
                (embedding, image_url)
            )
            result = cur.fetchone()
            if result is None:
                conn.rollback()
                return -1, "Lỗi khi đăng ký: Không thể tạo bản ghi mới trong database (bước 1)."
            new_id = result[0]

            # Bước 2: Cập nhật tên của bản ghi mới bằng chính ID của nó
            cur.execute(
                "UPDATE identities SET name = %s WHERE id = %s;",
                (str(new_id), new_id)
            )

            conn.commit()
            return new_id, f"Đăng ký thành công người dùng với ID: {new_id}."
        except Exception as e:
            conn.rollback()
            return -1, f"Lỗi khi đăng ký: {e}"
        finally:
            conn.close()

def recognize_person_db(embedding: np.ndarray, threshold: float = 0.6):
    """Tìm kiếm người dùng trong DB dựa trên embedding."""
    conn = get_db_connection()
    if conn is None:
        return "Unknown", -1.0

    register_vector(conn)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # `<->` là toán tử tính khoảng cách cosine (0=giống hệt, 2=khác biệt)
        # `1 - distance` sẽ là điểm tương đồng (similarity)
        cur.execute(
            """
            WITH distances AS (
                SELECT name, 1 - (embedding <-> %(emb)s) AS similarity
                FROM identities
            )
            SELECT name, similarity
            FROM distances
            ORDER BY similarity DESC
            LIMIT 1;
            """,
            {'emb': embedding}
        )
        result = cur.fetchone()

        try:
            if result and result['similarity'] >= threshold:
                # Ghi log nhận dạng thành công vào bảng recognition_logs
                try:
                    cur.execute(
                        """
                        INSERT INTO recognition_logs (identity_name, similarity)
                        VALUES (%s, %s);
                        """,
                        (result['name'], result['similarity'])
                    )
                    conn.commit()
                except Exception as e:
                    print(f"Lỗi khi ghi log nhận dạng: {e}")
                    conn.rollback()
                return result['name'], result['similarity']
            return "Unknown", (result['similarity'] if result else -1.0)
        finally:
            conn.close()

def update_person_image_url_db(person_id: int, image_url: str):
    """Cập nhật image_url cho một người dùng dựa trên ID."""
    conn = get_db_connection()
    if conn is None:
        return f"Lỗi: Không thể kết nối đến database."
    
    with conn.cursor() as cur:
        try:
            cur.execute(
                "UPDATE identities SET image_url = %s WHERE id = %s;",
                (image_url, person_id)
            )
            conn.commit()
            return f"Cập nhật URL ảnh thành công cho ID {person_id}."
        except Exception as e:
            conn.rollback()
            return f"Lỗi khi cập nhật URL ảnh cho ID {person_id}: {e}"
        finally:
            conn.close()

def update_person_image_url_db(person_id: int, image_url: str):
    """Cập nhật image_url cho một người dùng dựa trên ID."""
    conn = get_db_connection()
    if conn is None:
        return f"Lỗi: Không thể kết nối đến database."
    
    with conn.cursor() as cur:
        try:
            cur.execute(
                "UPDATE identities SET image_url = %s WHERE id = %s;",
                (image_url, person_id)
            )
            conn.commit()
            return f"Cập nhật URL ảnh thành công cho ID {person_id}."
        except Exception as e:
            conn.rollback()
            return f"Lỗi khi cập nhật URL ảnh cho ID {person_id}: {e}"
        finally:
            conn.close()

def update_person_name_db(person_id: int, new_name: str):
    """Cập nhật tên cho một người dùng dựa trên ID."""
    conn = get_db_connection()
    if conn is None:
        return f"Lỗi: Không thể kết nối đến database."
    
    with conn.cursor() as cur:
        try:
            cur.execute(
                "UPDATE identities SET name = %s WHERE id = %s;",
                (new_name, person_id)
            )
            if cur.rowcount == 0:
                return f"Không tìm thấy người dùng với ID '{person_id}' để cập nhật."
            conn.commit()
            return f"Đã cập nhật thành công tên cho người dùng ID {person_id} thành '{new_name}'."
        except Exception as e:
            conn.rollback()
            # Bắt lỗi tên bị trùng
            if "unique constraint" in str(e):
                return f"Lỗi: Tên '{new_name}' đã tồn tại. Vui lòng chọn tên khác."
            return f"Lỗi khi cập nhật tên cho ID {person_id}: {e}"
        finally:
            conn.close()

def delete_person_db(person_id: int):
    """Xóa một người dùng khỏi DB dựa trên ID."""
    conn = get_db_connection()
    if conn is None:
        return f"Lỗi: Không thể kết nối đến database."
        
    with conn.cursor() as cur:
        try:
            cur.execute("DELETE FROM identities WHERE id = %s;", (person_id,))
            # Kiểm tra xem có hàng nào bị xóa không
            if cur.rowcount == 0:
                return f"Không tìm thấy người dùng với ID '{person_id}' để xóa."
            conn.commit()
            return f"Đã xóa thành công người dùng với ID '{person_id}'."
        except Exception as e:
            conn.rollback()
            return f"Lỗi khi xóa ID '{person_id}': {e}"
        finally:
            conn.close()

def list_people_db():
    """Lấy danh sách tên tất cả người dùng trong DB."""
    conn = get_db_connection()
    if conn is None: return []
    with conn.cursor() as cur:
        cur.execute("SELECT id, name, image_url FROM identities ORDER BY id;")
        results = cur.fetchall()
    conn.close()
    return [{"id": row[0], "name": row[1], "image_url": row[2]} for row in results]

if __name__ == '__main__':
    print("Chạy thiết lập database...")
    setup_database()