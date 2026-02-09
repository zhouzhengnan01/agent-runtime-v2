import os


def get_postgres_driver() -> str:
    env_driver = os.getenv("POSTGRES_DRIVER") or os.getenv("DB_DRIVER")
    if env_driver:
        return env_driver.strip()
    for driver in ("psycopg2", "psycopg"):
        try:
            __import__(driver)
            return driver
        except Exception:
            continue
    return "psycopg2"
