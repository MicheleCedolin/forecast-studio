import sys

from .cli import main

# Koruma ZORUNLU: Windows'ta çok süreçli çalışma 'spawn' ile ana modülü yeniden içe aktarır; korumasız olsa her işçi
# komut satırını yeniden çalıştırırdı.
if __name__ == "__main__":
    sys.exit(main())
