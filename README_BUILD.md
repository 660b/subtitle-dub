# Sesli Windows build

Bu dosya, projenin Windows için tek tıklamayla çalıştırılabilir bir uygulamaya dönüştürülmesini anlatır.

## Gereksinimler

- Windows 10/11
- Python ortamı mevcut
- internet erişimi

## Kurulum

1. Terminali proje klasöründe açın.
2. Aşağıdaki komutu çalıştırın:

```powershell
Set-ExecutionPolicy -Scope Process -RemoteSigned
.\build_exe.ps1
```

3. İşlem tamamlandığında `dist\Sesli\` klasöründe EXE dosyası oluşur.

## Notlar

- Uygulama geliştirme modunda değil, paketlenmiş masaüstü uygulaması gibi çalışır.
- Büyük modeller ve dublaj için kullanıcı bilgisayarında uygun donanım gerekir.
- Hafif mod varsayılan olarak önerilir.
- İsterseniz daha sonra MSI/NSIS installer tarafı da eklenebilir.
