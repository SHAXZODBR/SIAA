# Sentinel Medical AI — klinikangiz uchun bosh miya MRT triaji

*SIAA Medical AI, Toshkent · dastur versiyasi 1.1.0 · bir sahifalik qisqacha tavsif*

## Bu nima

Sentinel Medical AI — klinika ichidagi bitta kompyuterga o‘rnatiladigan, shifokorga qaror qabul qilishda yordam beruvchi dastur. U bosh miya MRT tekshiruvlarini tahlil qiladi, patologiya bo‘lishi mumkin bo‘lgan tekshiruvlarni birinchi navbatda o‘qilishi uchun belgilaydi va shifokor tahrir qiladigan hisobot qoralamasini tayyorlaydi. **Har bir tekshiruvni rentgenolog shifokor o‘qiydi va imzolaydi.** Dastur O‘zbekiston Respublikasi Sog‘liqni saqlash vazirligi yoki boshqa biror nazorat organi tomonidan tibbiy buyum sifatida sertifikatlanmagan; undan klinikaning o‘z klinik boshqaruvi doirasida foydalaniladi.

## Bugun nima qiladi

| Funksiya | Siz uchun ma’nosi |
|---|---|
| **Triaj belgisi — bosh miya MRT (tasdiqlangan)** | O‘qitishda ishlatilmagan **178 ta ajratib qo‘yilgan bemorda** (Toshkent klinikasi) sinovdan o‘tkazilgan: **sezgirlik 0,90** (131 ta patologik tekshiruvdan 118 tasi belgilangan) va **spetsifiklik 0,47** (47 ta normal tekshiruvdan 22 tasi belgilanmagan; 95 % ishonch oralig‘i 0,33–0,61). Oddiy so‘z bilan: har 10 ta patologik tekshiruvdan taxminan 1 tasi belgilanmaydi, normal tekshiruvlarning esa taxminan yarmi **belgilanadi**. Shu klinikaning yangi bemorlarida keyinroq o‘tkazilgan tekshiruvda har uchta normal tekshiruvdan taxminan ikkitasi belgilangan. Ushbu tekshiruvda belgi asosan bemor yoshiga bog‘liq bo‘ldi: 40 yoshdan boshlab xulosadan qat’i nazar deyarli har bir tekshiruv belgilangan. Belgi «bu tekshiruvni avvalroq o‘qing» degani — bu tashxis emas. |
| **Uch til** | Interfeys va hisobot qoralamalari rus, o‘zbek (lotin yozuvi) va ingliz tillarida. |
| **Internetsiz** | Hammasi klinika kompyuterida internetsiz ishlaydi. Hech qanday tasvir, hisobot yoki bemor identifikatori klinikadan tashqariga chiqmaydi. |
| **PACS** | Tekshiruvlar klinika PACS tizimidan mahalliy Orthanc tuguni orqali avtomatik kelishi (ixtiyoriy) yoki papkadan yuklanishi mumkin; imzolangan hisobot tekshiruvga DICOM hujjat sifatida qaytarilishi mumkin. |
| **Imzolangan PDF** | Imzolashda shifokor, sana va vaqt hamda aniq matnning SHA-256 izi qayd etiladi. PDF da imzo bloki va AI natijasini bergan model identifikatori bor. |

## Nima qilmaydi

- Tashxis qo‘ymaydi va o‘zgarishlar joylashuvini aniqlamaydi.
- Hech qachon tekshiruvni norma deb tasdiqlamaydi: «belgilanmagan» — norma xulosasi emas, shifokor har bir tekshiruvni to‘liq o‘qiydi.
- Ushbu relizda o‘sma tashxisi yo‘q: ommaviy o‘sma klassifikatori ortiqcha belgilagani uchun sukut bo‘yicha o‘chirilgan va u uchun hech qanday ko‘rsatkich da’vo qilinmaydi.
- KT uchun ko‘rsatkich da’vo qilinmaydi: bosh miya KT tekshiruvi «AI tomonidan tahlil qilinmagan» deb ko‘rsatiladi va odatdagidek o‘qiladi. Shoshilinch qarorlar (o‘tkir insult yoki qon quyilishini istisno qilish), bolalar va operatsiyadan keyingi miya uchun mo‘ljallanmagan.

## Klinikaga nima kerak

- Bitta kompyuter: Windows 10/11 64-bit, kamida 4 yadroli protsessor (8 tavsiya etiladi), kamida 8 GB operativ xotira (16 GB tavsiya etiladi), SSD da kamida 20 GB bo‘sh joy (50 GB tavsiya etiladi), 1600×1000 ekran. Videokarta va internet kerak emas.
- Tekshiruvlarga kirish: PACS ulanishi yoki DICOM eksport papkasi.
- O‘rnatish uchun IT administratoringizning yarim kuni va rentgenologlar uchun bitta 60 daqiqalik o‘quv mashg‘uloti.

## Pilot klinikaga nima beradi

- Ustuvorlik berilgan ish ro‘yxati: belgilangan tekshiruvlar birinchi o‘qiladi.
- O‘z bemorlaringizda o‘lchangan ko‘rsatkichlar: rentgenologlaringiz baholagan kamida 100 ta ketma-ket tekshiruvda (shundan kamida 40 tasi normal) 95 % ishonch oralig‘i bilan sezgirlik va spetsifiklik.
- Pilot bo‘yicha yozma hisobot (rentgenologlaringiz bilan muvofiqlik, o‘qish vaqti, tizimning ishlash vaqti, xavfsizlik hodisalari) — o‘z qaroringiz uchun. Muntazam foydalanish faqat kelishilgan muvaffaqiyat mezonlari bajarilganda.

## Aloqa

SIAA Medical AI · Toshkent · www.siaa.uz
Mas’ul shaxs: [F.I.Sh., lavozimi] · Telefon: [+998 __ ___ __ __] · Elektron pochta: [manzil]
