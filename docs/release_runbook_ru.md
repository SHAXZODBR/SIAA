# Регламент релиза — дообучение модели триажа мозга на новой партии данных клиники

Экземпляр для основателя / ML-инженера. Английская версия: `release_runbook_en.md`. Контракт, по
которому написан каждый этап: `dev/release_pipeline_contract.md`. Оркестратор:
`scripts/release_pipeline.py`.

Цель: новая партия МРТ / КТ головного мозга **с заключениями рентгенологов** приезжает на диске →
одна возобновляемая команда обучает кандидата, проверяет его на зафиксированных отложенных
пациентах, продвигает в релиз, переписывает все закреплённые числа в коде и документации и
публикует бандл модели → вы ставите тег приложения, CI собирает установщик, клиника его получает.
Ничего из `data/` и `models/` никогда не коммитится; имена пациентов не попадают ни в один
выходной файл.

## 1. Что покрывает этот регламент

| | |
|---|---|
| Вход | корень диска с одной папкой на исследование (DICOM) + заключения к этим исследованиям |
| Выход | `models/brain_triage_finetuned/` (новая идентичность), обновлённые документы/код, бандл в `data/release/<tag>/`, GitHub-релиз модели, затем установщик, собранный CI |
| Кто | один человек на обучающем Mac (Apple Silicon, MPS); ~1 рабочий день реального времени на 5 000 исследований, в основном без присмотра |
| Никогда | `git push`, `git tag` и секрет GitHub скрипт **не трогает** — это ваши ручные шаги в разделе 8 |

## 2. Предварительные условия

1. **Диск смонтирован** и читается, например `/Volumes/Transcend/new1`. Одна папка на исследование; индексатор читает один DICOM-заголовок на папку. Проверка: `ls /Volumes/Transcend/new1 | head`.
2. **Заключения в одном из трёх принимаемых форматов** (все разбирает `scripts/parse_hospital_reports.py`):
   - папка PDF по пациентам с именами файлов `<patientID>_<Имя>.pdf` (формат Shox International Hospital; один PDF может содержать несколько исследований, каждое под заголовком `МРТ головного мозга (ГГГГ-ММ-ДД)` / `МСКТ …` и строкой `ЗАКЛЮЧЕНИЕ:`), **или**
   - заполненный **лист запроса** `.xlsx`/`.csv` в формате `data/new1_report_request_Shox_2026-06-27_to_2026-09-15.xlsx` — русские заголовки `ID пациента (PatientID)`, `Дата исследования` (ДД.ММ.ГГГГ), `Модальность`, `Область`, `Описание исследования`, `Заключение врача (вставить сюда)`; клиника вставляет заключение в последний столбец, **или**
   - `.docx`-экспорт того же.
   Скопируйте заключения в локальную папку (например `~/Desktop/reports_<batch>`); никогда — в репозиторий.
3. **Python venv**: `/Users/shakhzodbtr/venv/bin/python` (3.14, torch 2.11, MPS). Проверка: `/Users/shakhzodbtr/venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"` → `True`.
4. **Диск**: ≥ 20 ГБ свободно на Mac (срезы ≈ 5 PNG на исследование, две копии модели, части бандла). Ниже этого порога конвейер отказывается обучать.
5. **Текущий релиз цел**: существуют `models/brain_triage_finetuned/{model.safetensors,MANIFEST.json,study_level_eval.json}` и `data/holdout_patients_v1.txt` (178 зафиксированных id пациентов).
6. **Чистое git-дерево** перед стартом (`git status`), чтобы diff этапа bump можно было проверить.
7. **Доступ для публикации**: токен GitHub, которым пользуется `git push` (keychain), или `SIAA_MODELS_TOKEN` в окружении; права записи в приватный репозиторий моделей `SHAXZODBR/SIAA-models` и в секрет окружения `MODEL_BUNDLE_URL` репозитория приложения.

## 3. Оценки времени (5 000 исследований, Apple Silicon)

| Этап | Ожидаемое время | Примечания |
|---|---|---|
| index | 10–30 мин | один DICOM-заголовок на папку исследования; возобновляемый |
| parse | минуты | ~5 000 PDF; xlsx/csv — секунды |
| label + benchmark | < 1 мин | только правила; LLM-проход (`--label-llm`) необязателен и медленнее |
| match | секунды | нужен индекс |
| build | ~1 исследование / с → ~1,5 ч | возобновляемый, переиспользует готовые PNG |
| combine | минуты | симлинки + манифест |
| train | ≈ 45 мин на 3 500 срезов → **≈ 5 ч** на 5 000 исследований (12 эпох) | MPS; Mac на питании, крышка открыта |
| gate | ~5 мин | кандидат + старая модель на одном и том же val |
| promote, bump | секунды | |
| bundle | 5–15 мин | zip, разбиение, загрузка, проверка восстановления |

## 4. Путь в одну команду

```sh
cd ~/Desktop/Siaa_ai/sentinel
/Users/shakhzodbtr/venv/bin/python scripts/release_pipeline.py plan --batch new1 \
    --root /Volumes/Transcend/new1 --reports ~/Desktop/reports_new1          # сначала прочитайте
/Users/shakhzodbtr/venv/bin/python scripts/release_pipeline.py run  --batch new1 \
    --root /Volumes/Transcend/new1 --reports ~/Desktop/reports_new1
```

- `plan` печатает shell, который можно скопировать и вставить (каждая команда, защитные проверки
  комментариями), и ничего не запускает.
- `run` выполняет одиннадцать этапов по порядку, пишет лог каждого в
  `data/release/<tag>/logs/<stage>.log`, вывод защитных проверок — в `data/release/<tag>/logs/guards.log`,
  и ведёт `data/release/<tag>/state.json` (все действующие параметры, статус / входы / выходы каждого
  этапа, вердикты проверок).
- **Тег фиксирован на весь релиз.** Первая напечатанная строка — `tag: <tag> (…)`. Первый `run`
  выбирает следующий свободный `brain-triage-v<N>` и записывает его в `data/release/LATEST`; каждый
  следующий запуск без `--tag` продолжает LATEST (`tag: <tag> (from LATEST)`), пока его этап `bundle` не
  выполнен — даже после того как `promote` создал новую папку `models/brain_triage_finetuned_prev_*`.
  `--tag` нужен только чтобы задать имя самому или вернуться к старому релизу.
- При первом сбое останавливается, печатает упавшую команду, путь к логу и точную строку
  `… run … --resume`, в которой повторены **все** параметры запуска (`--epochs`, `--device`,
  `--python`, `--base-model`, `--llm-labels`, …). Устраните причину и вставьте эту строку. Короткая
  форма `/Users/shakhzodbtr/venv/bin/python scripts/release_pipeline.py run --batch new1 --resume`
  делает то же самое: тег из LATEST, параметры из `state.json` (печатаются как `options from state.json: …`).
- `--dry-run` печатает команды, ничего не запуская; `--from <stage>` / `--only <stage>` (или просто
  `release_pipeline.py <stage> …`) сужают запуск; `status` показывает таблицу состояния.
- Старые наборы объединяются автоматически: `data/local_finetune_bin` (партия 1) плюс все
  `data/sets/*`, кроме текущей партии и репетиционных наборов (`data/sets/rehearsal_*`), от старых к
  новым; текущая партия идёт **последней**, потому что `combine` даёт каждому пациенту ОДИН класс, и
  при конфликте меток побеждает последний набор в командной строке (конфликты печатаются и
  записываются в `dataset_manifest.json`). Переопределяется через `--extra-sets`.
- Пациенты, на которых обучена текущая модель, остаются в `train`: `train/` старого набора всегда, а
  начиная со второго переобученного релиза набор, указанный в `training_metrics.json` текущей модели,
  автоматически передаётся как `--train-lock data/train_release_<предыдущий тег>` (`plan` его
  печатает). `--train-lock <ids.txt | папка релизного набора>` задаёт его вручную.

## 5. По этапам — что печатает каждый и как выглядит «хорошо»

| # | Этап | Скрипт | Печатает | «Хорошо» выглядит так |
|---|---|---|---|---|
| 0 | `index` | `scripts/index_new_studies.py --root … --out data/<batch>_index.csv` | найденные исследования, разбивка MR/CT, ошибки | строк ≈ папок исследований; столбец `error` в основном пуст; число МРТ мозга + КТ головы правдоподобно |
| 1 | `parse` | `scripts/parse_hospital_reports.py --src … --out data/<batch>_reports.csv` | прочитанные файлы, найденные секции, счётчики по модальностям | одна строка на **секцию** исследования; модальность из заголовка секции (есть и MR, и CT, а не всё МСКТ); в `conclusion` никогда нет имени |
| 2 | `label` | `scripts/label_reports.py --reports … --out data/<batch>_labels.csv [--import-llm-labels <jsonl>]`, затем `… --benchmark` | распределение 9 классов + abnormal, доля `needs_review`; согласие бенчмарка + матрица ошибок против эталона на 894 пациента | согласие бенчмарка по `abnormal` ≥ 0,90 (иначе ненулевой код выхода); `needs_review` < 15 %; заметная доля `normal` |
| 3 | `match` | `scripts/match_images_to_reports.py --images … --index … --labels … --out data/<batch>_matched.csv` | сопоставленные / несопоставленные исследования мозга, гистограмма `match_quality` | большинство исследований мозга сопоставлено; качество в основном «тот же id + та же дата»; немного «единственное заключение пациента» |
| 4 | `build` | `scripts/build_training_set.py … --out data/sets/<batch> --mode binary --abnormal-col abnormal --skip-needs-review` | записанные срезы по классам, `studies.csv` | ≈ 5 срезов на исследование; строк в `studies.csv` = сопоставленные исследования мозга минус `needs_review` |
| 5 | `combine` | `scripts/combine_training_sets.py --sets data/local_finetune_bin … data/sets/<batch> --holdout data/holdout_patients_v1.txt [--train-lock …] --out data/train_release_<tag>` | счётчики по партиям, классы по сплитам (train / val / tune), конфликты меток (побеждает последний набор), число закреплённых за train, проверка holdout | **0** пересечений пациентов train/val/tune; все 178 отложенных пациентов в `val`; один класс на пациента; есть `tune/`; записан `dataset_manifest.json` |
| 6 | `train` | `training/finetune_brain_classifier.py … --base-model models/brain_triage_finetuned --epochs 12 --device mps --select-split tune --output models/brain_triage_candidate_<tag>` | loss + accuracy по эпохам; лучшая эпоха выбирается по `tune/`, никогда по `val` (набору гейта) | loss падает и выходит на плато; нет NaN; записан `training_metrics.json` с `checkpoint_selection.split = tune` |
| 7 | `gate` | `scripts/eval_study_level.py --model <candidate> --data data/train_release_<tag>/val --write-manifest --min-sens 0.85 --min-spec 0.40 --compare-with models/brain_triage_finetuned` | чувствительность/специфичность на уровне исследования с 95 % ДИ для кандидата и старой модели, вердикт `promotion`, покрытие holdout | гейт **пройден**, `promotion.promote = true`, `protocol` = продакшн (среднее по 5 центральным срезам @0,50, предиктор реестра), `holdout_covered = true`, `MANIFEST.json` записан рядом с кандидатом |
| 8 | `promote` | `scripts/promote_model.py --candidate … --dest models/brain_triage_finetuned` | новая идентичность (12 hex), куда ушёл старый релиз | старый релиз в `models/brain_triage_finetuned_prev_<old_id>/`; реестр загружает новую папку |
| 9 | `bump` | `scripts/bump_model_release.py --model models/brain_triage_finetuned` | список переписанных файлов | все закреплённые числа, идентичность и даты обновлены; `--check` чист |
| 10 | `bundle` | `packaging/publish_model_bundle.py --model … --out data/release/<tag>/ --github SHAXZODBR/SIAA-models --tag <tag>` | части, sha256, URL релиза, **API-URL ассета манифеста**, проверка восстановления | `manifest.json` (`validated: true` + сводка `eval`) + части < 100 МБ; восстановление во временную папку проверено |

Защитные проверки оркестратора **перед `train`**: бенчмарк разметки пройден (или
`--skip-benchmark`), напечатаны число сопоставленных исследований мозга и число нормы
(предупреждение ниже 300 нормальных), `dataset_manifest.json` есть, пересечений пациентов
train/val/tune нет и каждый holdout-id в `val`, есть `tune/` (тренер запускается с `--select-split tune`;
без `tune/` релиз останавливается), ≥ 20 ГБ свободно, диск смонтирован. **Перед `bundle`**: гейт
пройден и promotion = true у продвинутой модели, плюс все правила `promote_model.py` (релизные пороги,
продакшн-протокол, релизный val / покрытие holdout, оценка привязана к этим весам),
`bump_model_release.py --check` чист, полный `pytest -q` и `docs/test_docs_consistency.py` зелёные
(это единственное место, где запускается весь набор тестов). Полный вывод проверок пишется в
`data/release/<tag>/logs/guards.log`, вердикт — в `state.json` (`guards.pre_train`, `guards.pre_bundle`).

## 6. Правила go / no-go

| Правило | Порог | Где проверяется |
|---|---|---|
| Порог чувствительности на уровне исследования | ≥ 0,85 | `eval_study_level.py --min-sens` (код 2) |
| Порог специфичности на уровне исследования | ≥ 0,40 | `eval_study_level.py --min-spec` (код 2) |
| Продвижение относительно текущего релиза | новая проходит гейт **и** balanced accuracy ≥ старая − 0,01 **и** чувствительность ≥ старая − 0,03 | `eval_study_level.py --compare-with` (код 4), `promote_model.py` |
| Зафиксированные отложенные пациенты | все 178 id из `data/holdout_patients_v1.txt` в `val`; ни одного в `train` | `combine_training_sets.py`, проверка оркестратора перед train |
| Минимум нормы в партии | предупреждение ниже 300 сопоставленных нормальных исследований — без них специфичность не сдвинется | проверка оркестратора перед train |
| Качество разметки | правила согласуются с эталоном на 894 пациента по `abnormal` ≥ 0,90 | `label_reports.py --benchmark` |
| Манифест | sha256 весов закреплён в `MANIFEST.json`, проверяется перед каждой оценкой | `eval_study_level.py` (код 3) |
| Релизный протокол | оценка выполнена с релизными порогами, продакшн-гейтом (`gate mean @0.50`, предобработка предиктора реестра), на релизном val (его sha256 пациентов = `dataset_manifest.json` → `splits.val.patients_sha256`; для старой папки val оценены все holdout-id — `eval_study_level.py --holdout`, по умолчанию `data/holdout_patients_v1.txt`, `none` — пропустить) и для этих весов | `promote_model.py` (отказ, код 1), проверка перед bundle, `publish_model_bundle.py` |

`promote_model.py --force` отменяет **только** `promotion.promote = false` (сравнение старой и новой
модели; записывается как `forced` в `promotion_log.json`, и `publish_model_bundle.py` затем отказывает
такой модели без `--allow-unvalidated`). Он никогда не отменяет проваленный гейт, пониженные пороги,
не продакшн-гейт или предобработку, неверный val / непокрытый holdout или оценку других весов — такие
отказы печатают `not overridable` и `[--force ignored]`. Для старой папки val `promote_model.py
--holdout` задаёт файл закреплённого holdout (по умолчанию `data/holdout_patients_v1.txt`).

Падение чувствительности никогда не обменивается на рост специфичности: если кандидат
специфичнее, но пропускает больше патологии, чем допускает правило, он **не** продвигается.

## 7. Что делать, если гейт не пройден

Этап `gate` останавливает запуск **до `promote`** двумя разными способами; в обоих случаях
`models/brain_triage_finetuned` не тронута, а `state.json` хранит вердикт (его показывает `status`):

| Код | Печатает | Значение |
|---|---|---|
| 2 | `GATE FAILED` (вердикт `gate_failed`) | кандидат ниже порога (sens ≥ 0,85 / spec ≥ 0,40) |
| 4 | `NOT PROMOTED` (вердикт `not_promoted`) | кандидат проходит пороги, но не лучше текущего релиза (правило продвижения, раздел 6) |

1. Прочитайте `models/brain_triage_candidate_<tag>/study_level_eval.json` и `data/release/<tag>/logs/gate.log`: какой порог не взят, насколько широк ДИ, `promotion.reason`.
2. **Оставьте старую модель** — это всегда допустимый исход. Ничего не продвинуто; у клиники остаётся текущий релиз. Папку кандидата удалите или сохраните; `data/release/<tag>/` фиксирует, что было испробовано.
3. **Слишком низкая специфичность** (обычный случай): в партии мало нормы. Запросите у клиники больше *нормальных* исследований мозга с заключениями (лист запроса из раздела 2), соберите новую партию, обучите снова. Порог не снижайте.
4. **Слишком низкая чувствительность**: сначала проверьте разметку — переразметьте сомнительные заключения LLM-workflow (команды ниже), затем посмотрите `results` в JSON оценки для агрегации `mean` при других порогах — смена порога это изменение **продукта** (`brain_analysis.py`), а не ручка, чтобы пройти гейт.
5. **Переобучение с публичной базы ViT** (запасной путь, если дообучение текущего релиза снова и снова не проходит): этап train стартует с `--base-model` (по умолчанию `models/brain_triage_finetuned`). Используйте офлайн-копию `models/hf/google__vit-base-patch16-224`, если она есть, иначе id `google/vit-base-patch16-224` (уже лежит в `~/.cache/huggingface` на этом Mac; сообщение о провале гейта печатает нужный вариант). Гейт по-прежнему сравнивает с текущим релизом. Вставьте строку, которую напечатал провал гейта, или:

   ```sh
   /Users/shakhzodbtr/venv/bin/python scripts/release_pipeline.py run --batch new1 --tag <tag> \
       --base-model google/vit-base-patch16-224 --from train
   ```
   `--from train` (без `--resume`) всегда переобучает; с `--resume` тоже, потому что `train` записывает свои `--base-model` и `--epochs`, и новое значение считается изменённым входом.
6. Никогда не дообучайте на горстке дополнительных исследований, чтобы протиснуться через порог — числа должны быть сопоставимы между релизами (тот же holdout, тот же протокол).

**Путь переразметки (LLM-метки), по шагам.** Запускайте из `~/Desktop/Siaa_ai/sentinel`; `<tag>` — тег, напечатанный запуском.

1. Выгрузите сомнительные строки (`needs_review=yes`; `--all` — все строки мозга). Команда заодно перезаписывает метки правил:

   ```sh
   /Users/shakhzodbtr/venv/bin/python scripts/label_reports.py --reports data/new1_reports.csv \
       --out data/new1_labels.csv --export-for-llm data/new1_review.jsonl
   ```
2. В Claude Code, в этом репозитории, запустите Workflow (он работает только внутри инструмента Workflow в Claude Code, не через `node`):

   ```
   Workflow({ scriptPath: 'training/relabel_workflow.js', args: { file: 'data/new1_review.jsonl' } })
   ```
3. Сохраните возвращённый массив `labels` как JSONL — по одному объекту `{"patient_id","study_date","modality","section_index","label","abnormal"}` в строке — в `data/new1_llm_labels.jsonl` (только внутри `data/`; имён в нём нет). Импорт сопоставляет по (пациент, дата, модальность, секция), а (пациент, дата, модальность) и (пациент, дата) использует, только если такой ключ указывает ровно на одну строку, и печатает `N applied, N ambiguous, N skipped`; неоднозначные метки не применяются.
4. Слейте метки:

   ```sh
   /Users/shakhzodbtr/venv/bin/python scripts/label_reports.py --reports data/new1_reports.csv \
       --out data/new1_labels.csv --import-llm-labels data/new1_llm_labels.jsonl
   ```
5. Продолжите релиз с `match`, записав JSONL в `state.json`:

   ```sh
   /Users/shakhzodbtr/venv/bin/python scripts/release_pipeline.py run --batch new1 --from match \
       --tag <tag> --llm-labels data/new1_llm_labels.jsonl --resume
   ```
   `match → build → combine → train → gate` выполняются заново, потому что изменились их входы (новые
   метки → новый `studies.csv` → новый `dataset_manifest.json`). `--fresh` не нужен: `build` переносит
   срезы переразмеченного исследования в папку нового класса и печатает `relabelled N`. Повторный запуск этапа `label` **без**
   `--import-llm-labels` заново создал бы метки только по правилам и потерял бы LLM-метки; когда
   `--llm-labels` записан в `state.json`, этап `label` оркестратора всегда передаёт
   `--import-llm-labels <jsonl>`, в том числе при последующем `--from label`.

## 8. Публикация: секрет, CI, установщик

После `bundle` оркестратор печатает этот список; сам он ничего из этого не делает.

1. **Секрет**: скопируйте *API-URL ассета манифеста* (`https://api.github.com/repos/SHAXZODBR/SIAA-models/releases/assets/<id>`) в окружение `MODEL_BUNDLE_URL` репозитория приложения → секрет `MODEL_BUNDLE_URL`; держите действительным `MODEL_BUNDLE_TOKEN` (токен с чтением приватного репозитория моделей).
2. **Тесты**: уже зелёные (их запустила проверка перед bundle); если что-то правили после, повторите `/Users/shakhzodbtr/venv/bin/python -m pytest -q && /Users/shakhzodbtr/venv/bin/python -m pytest -q docs/test_docs_consistency.py`.
3. **Закоммитьте bump**: `git status`, `git diff` — меняться должны только файлы кода/документации из списка этапа bump; `git add` именно их (никогда `data/`, никогда `models/`), коммит `release: <tag> — bump validated numbers`.
4. **Версия приложения — в каждом релизе**: поднимайте `"version"` в `desktop-app/package.json` при **каждом** релизе, в том числе только с новой моделью (минорная версия, например `1.0.0 → 1.1.0`), чтобы тег `v<version>` был новым — `git tag` отказывает на существующем теге, а push старого тега не запускает CI. Закоммитьте.
5. **Тег и push**: `git tag v<version> && git push origin main && git push origin v<version>`. Тег запускает `.github/workflows/build-windows.yml` («Build installers»); он восстанавливает модель из `MODEL_BUNDLE_URL` и собирает NSIS-установщик (и macOS DMG). Его шаг «Verify bundle identity matches the code» останавливает сборку, если идентичность восстановленной модели отличается от `src/inference/validated_metrics.FALLBACK_IDENTITY`, — поэтому закоммитьте bump и обновите секрет **до** тега.
6. **Скачайте установщик** из артефактов запуска Actions (или из черновика GitHub-релиза, который workflow создаёт для тегов `v*`).
7. **Тест на ноутбуке**: пройдите `windows_laptop_test_ru.md` (EN: `windows_laptop_test_en.md`) на Windows-ноутбуке с выключенным Wi-Fi; все пункты отмечены; в «О программе» и в подвале PDF **новая** идентичность.
8. **Опубликуйте черновик релиза** с описанием: новые чувствительность / специфичность с ДИ, число отложенных пациентов, идентичность модели, имя партии.
9. Передайте установщик + `admin_install_guide_*.md` ИТ-службе клиники; храните `data/release/<tag>/`, пока установка не подтверждена.

## 9. Возобновление, статус и устранение неполадок

- `/Users/shakhzodbtr/venv/bin/python scripts/release_pipeline.py status --batch <batch>` — таблица этапов (pending / running / done / archived / failed) с путями к логам, вердиктами проверок и вердиктом гейта; тег берётся из `data/release/LATEST`.
- `--resume` пропускает этап со статусом `done`, чьи выходы на месте и входы не изменились (sha256 входных файлов: `studies.csv` каждого объединяемого набора — для старого набора без него отсортированный список файлов, — файлы holdout и train-lock, `dataset_manifest.json`, веса кандидата, JSONL с LLM-метками; скрытые файлы вроде `.DS_Store` из Finder не учитываются; для `train` также значения `--base-model` и `--epochs`); иначе этап выполняется заново.
- `promote` **перемещает** папку кандидата в `models/brain_triage_finetuned`. Он записывает `archived_candidate` в `state.json` и помечает `train` и `gate` как `archived` (= выполнено), поэтому возобновление после promote выполняет только оставшееся (`bump`, `bundle`). Этап, чьих выходов уже нет, тоже считается выполненным, если выполнен более поздний этап, — если только его не называет `--from` / `--only`. После promote этапы `combine` / `train` / `gate` / `promote` под тем же тегом больше не запускаются (гейт сравнил бы нового кандидата с самим собой; combine переписал бы val-набор, к которому привязана оценка продвинутой модели), а `--resume` считает все этапы до `promote` замороженными, даже если какой-то входной файл с тех пор изменился, — остаются только `bump` и `bundle`. Изменения после promote идут в следующий релиз (новый тег).
- Train-lock: `combine` закрепляет в `train` пациентов, на которых обучена текущая модель релиза (`data/train_release_<предыдущий тег>`, из её `training_metrics.json`). Сохраняйте эту папку. Если её нет или она пересобрана, проверка перед `combine` останавливает запуск; восстановите папку или передайте `--train-lock <ids.txt>` со списком id пациентов, на которых обучена эта модель.
- Выбор контрольной точки: `combine` пишет сплит `tune/` (часть новых пациентов train); тренер запускается с `--select-split tune` и выбирает лучшую эпоху по `tune/`, никогда по `val` (набору гейта). Проверка перед train останавливает релиз, если `tune/` нет (перезапустите `combine` с `--tune-frac > 0`).
- `bundle` отказывает (код 1) → либо модель не валидирована (`publish_model_bundle.py` проверяет те же факты гейта; `--allow-unvalidated` — только для демо, пишет `validated: false` в `manifest.json`), либо GitHub-релиз с этим тегом уже содержит бандл **другой** идентичности модели (`--replace-existing` перезаписывает его намеренно). Повторный запуск с тем же бандлом переиспользует загруженные ассеты, поэтому id ассета `MODEL_BUNDLE_URL` не меняется; `--reupload` загружает заново (новый id ассета → вставьте секрет снова). Без `--tag` `publish_model_bundle.py` использует тег `brain-triage-<identity12>`; оркестратор всегда передаёт тег релиза.
- Диск отмонтировался посреди запуска → падают `index`/`match`/`build`; смонтируйте, `--resume`.
- Бенчмарк ниже 0,90 → правила разметки деградировали; исправьте `label_reports.py`, повторите `label` (`--from label --resume`; импортированные LLM-метки сохраняются, если `--llm-labels` записан).
- `combine` падает на пересечении → один id пациента оказался в двух наборах по разные стороны; везде должен быть один и тот же числовой id — сверьте `studies.csv` обоих наборов.
- `promote` отказывает → реестр приложения не смог загрузить папку кандидата; проверьте наличие `config.json` / `preprocessor_config.json` и `classes = ['abnormal','normal']`.
- `bump --check` не чист → выполните этап `bump` ещё раз, затем посмотрите `git diff`.
- Всё промежуточное лежит в `data/release/<tag>/`; чтобы бросить релиз, запустите с новым `--tag` (готовые срезы в `data/sets/` переиспользуются).

## 10. Итоговый чек-лист релиза

Пути подтверждений указаны относительно `data/release/<tag>/`, если не начинаются с `data/`, `models/` или URL.

| ☐ | Пункт | Подтверждение |
|---|---|---|
| ☐ | Заключения разобраны, модальность из заголовков секций, имён нет ни в одном CSV | `data/<batch>_reports.csv`, `logs/parse.log` |
| ☐ | Бенчмарк разметки ≥ 0,90 по `abnormal` | `logs/label.log`, `state.json` → `guards.label_benchmark` |
| ☐ | ≥ 300 нормальных исследований сопоставлено (или задокументированная причина продолжать) | `state.json` → `guards.matched_counts` / `guards.pre_train`, `logs/guards.log` |
| ☐ | 0 пересечений train/val/tune, 178 holdout-id в val | `data/train_release_<tag>/dataset_manifest.json`, `logs/guards.log` |
| ☐ | Гейт пройден: sens ≥ 0,85, spec ≥ 0,40, promotion true | `models/brain_triage_finetuned/study_level_eval.json` |
| ☐ | Новая идентичность напечатана и показывается приложением | `MANIFEST.json`, `/health`, подвал PDF |
| ☐ | Bump применён, `bump_model_release.py --check` чист | `logs/guard_bump_check.log`, `logs/guards.log` |
| ☐ | `pytest -q` и проверка документации зелёные | `logs/guard_pytest*.log`, `state.json` → `guards.pre_bundle` |
| ☐ | Бандл опубликован, восстановление проверено | `manifest.json`, `logs/bundle.log` |
| ☐ | Секрет `MODEL_BUNDLE_URL` обновлён | GitHub → Settings → Environments |
| ☐ | Версия в `desktop-app/package.json` поднята, bump закоммичен, новый `v<version>` протегирован и запушен | `git log`, зелёный запуск Actions |
| ☐ | Установщик скачан, тест на ноутбуке офлайн пройден | отмечен `windows_laptop_test_*.md` |
| ☐ | Черновик релиза опубликован с числами + идентичностью | страница GitHub-релиза |
| ☐ | Старый релиз сохранён в `models/brain_triage_finetuned_prev_<id>/` для отката | `ls models/` |
