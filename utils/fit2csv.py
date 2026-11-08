import csv
import os
import fitparse
import pytz

allowed_fields = ['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'exception', 'battery', 'spo2']
required_fields = ['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'exception', 'battery', 'spo2']

UTC = pytz.UTC
CST = pytz.timezone('Europe/Vienna')


def main():

    output_dir = "data/cardiac/csv"
    os.makedirs(output_dir, exist_ok=True)
    source_dir = "."

    files = os.listdir(source_dir)
    fit_files = [file for file in files if file[-4:].lower() == '.fit']

    for file in fit_files:
        fit_path = os.path.join(source_dir, file)
        new_filename = os.path.join(output_dir, file[:-4] + '.csv')

        if os.path.exists(new_filename):
            # print('%s already exists. skipping.' % new_filename)
            continue

        fitfile = fitparse.FitFile(fit_path, data_processor=fitparse.StandardUnitsDataProcessor())

        print('converting %s -> %s' % (file, new_filename))
        write_fitfile_to_csv(fitfile, new_filename)

    print('finished conversions')


def write_fitfile_to_csv(fitfile, output_file='test_output.csv'):
    messages = fitfile.messages
    data = []
    for m in messages:
        skip = False
        if not hasattr(m, 'fields'):
            continue
        fields = m.fields
        # check for important data types
        mdata = {}
        for field in fields:
            if field.name in allowed_fields:
                if field.name == 'timestamp':
                    mdata[field.name] = UTC.localize(field.value).astimezone(CST)
                else:
                    mdata[field.name] = field.value
        for rf in required_fields:
            if rf not in mdata:
                skip = True
                break
        if not skip:
            data.append(mdata)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(allowed_fields)
        for entry in data:
            writer.writerow([str(entry.get(k, '')) for k in allowed_fields])
    print('wrote %s' % output_file)


if __name__ == '__main__':
    main()