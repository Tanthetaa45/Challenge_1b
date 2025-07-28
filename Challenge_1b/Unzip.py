import zipfile, pathlib
zip_path = pathlib.Path('Challenge_1b/submission_1b.zip')
with zipfile.ZipFile(zip_path) as z:
    z.extractall('Challenge_1b/extracted_submission')
