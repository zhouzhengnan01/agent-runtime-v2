"""
数据库迁移脚本：添加json_schema字段到agents表
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from app.db.session import SessionLocal
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)

def add_json_schema_field():
    """添加json_schema字段到agents表"""
    db = SessionLocal()

    try:
        print("🔄 开始数据库迁移：添加json_schema字段...")

        # 1. 检查字段是否已存在
        check_result = db.execute(text("""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
            AND TABLE_NAME = 'agents'
            AND COLUMN_NAME = 'json_schema'
        """))

        if check_result.fetchone():
            print("✅ json_schema字段已存在，无需迁移")
            return

        # 2. 添加字段
        print("📝 添加json_schema字段...")
        alter_sql = text("""
            ALTER TABLE agents
            ADD COLUMN json_schema JSON
            COMMENT 'JSON输出格式定义，仅在output_format=\'json\'时生效'
        """)

        db.execute(alter_sql)
        db.commit()

        print("✅ 成功添加json_schema字段")

        # 3. 验证字段
        verify_result = db.execute(text("""
            DESCRIBE agents
        """))

        print("📊 更新后的agents表结构:")
        for row in verify_result.fetchall():
            if row and len(row) > 0:
                field_name = row[0] if row[0] else 'N/A'
                print(f"  {field_name:30s}")

        print("🎉 数据库迁移完成！")

    except Exception as e:
        print(f"❌ 数据库迁移失败: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def create_migration_example():
    """创建一些示例JSON格式配置"""
    db = SessionLocal()

    try:
        print("\n💡 创建JSON格式示例配置...")

        # 为一些现有智能体添加JSON格式配置示例
        examples = [
            {
                "response_type": "structured_response",
                "format": "json",
                "fields": [
                    {
                        "name": "answer",
                        "type": "string",
                        "description": "回答内容"
                    },
                    {
                        "name": "confidence",
                        "type": "number",
                        "description": "置信度(0-1)"
                    }
                ]
            },
            {
                "response_type": "data_analysis",
                "format": "json",
                "fields": [
                    {
                        "name": "summary",
                        "type": "string",
                        "description": "分析总结"
                    },
                    {
                        "name": "key_points",
                        "type": "array",
                        "description": "关键要点"
                    }
                ]
            }
        ]

        # 更新前几个智能体作为示例
        sample_agents = db.execute(text("""
            SELECT id, name, output_format
            FROM agents
            WHERE status = 'active'
            LIMIT 2
        """)).fetchall()

        for i, agent in enumerate(sample_agents):
            if i < len(examples):
                # 使用json_schema字段存储配置
                update_sql = text("""
                    UPDATE agents
                    SET json_schema = :schema,
                        output_format = 'json',
                        updated_at = NOW()
                    WHERE id = :id
                """)

                db.execute(update_sql, {
                    'schema': examples[i],
                    'id': agent[0]
                })

                print(f"  📝 {agent[1]} ({agent[0][:8]}...) → 设置为JSON格式")

        db.commit()
        print("✅ 示例配置创建完成")

    except Exception as e:
        print(f"❌ 创建示例配置失败: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("🚀 数据库迁移工具")
    print("=" * 50)

    # 执行迁移
    add_json_schema_field()

    # 创建示例
    create_migration_example()

    print("\n🎉 迁移完成！现在可以使用json_schema字段了。")