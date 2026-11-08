import csv
import os
import fitparse
import pytz

#new miky
# allowed_fields = ['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'spo2', 'battery', 'gyrox', 'gyroy','gyroz']#['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'exception','ds']
# required_fields = ['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'spo2', 'battery', 'gyrox', 'gyroy','gyroz'] 
allowed_fields = ['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'exception', 'battery','spo2']#['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'exception','ds']
required_fields = ['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'exception', 'battery','spo2'] 

# Our old
# allowed_fields = ['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'exception']#['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'exception','ds']
# required_fields = ['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'exception'] 
# allowed_fields = ['timestamp', 'heart_rate'] 
# required_fields = ['timestamp', 'heart_rate'] 
# Juventus
# allowed_fields = ['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'exception', 'battery']#['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'exception','ds']
# required_fields = ['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'exception', 'battery'] 
# Spo2
# allowed_fields = ['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'exception', 'battery', 'spo2']#['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'exception','ds']
# required_fields = ['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'exception', 'battery', 'spo2'] 
#Tixi Altitude
# allowed_fields = ['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'exception', 'enhanced_altitude']#['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'exception','ds']
# required_fields = ['timestamp', 'hr', 'hrv', 'accx', 'accy', 'accz', 'exception', 'enhanced_altitude'] 
# Tixi Exercise
# allowed_fields = ['timestamp', 'training_load_peak']
# required_fields = ['timestamp', 'training_load_peak']

UTC = pytz.UTC
CST = pytz.timezone('Europe/Vienna')


def main():
    files = os.listdir()
    fit_files = [file for file in files if file[-4:].lower()=='.fit']
    for file in fit_files:
        new_filename = file[:-4] + '.csv'
        if os.path.exists(new_filename):
            #print('%s already exists. skipping.' % new_filename)
            continue
        fitfile = fitparse.FitFile(file,  
            data_processor=fitparse.StandardUnitsDataProcessor())
        
        print('converting %s' % file)
        write_fitfile_to_csv(fitfile, new_filename)
    print('finished conversions')

def write_fitfile_to_csv(fitfile, output_file='test_output.csv'):
    messages = fitfile.messages
    data = []
    for m in messages:
        skip=False
        if not hasattr(m, 'fields'):
            continue
        fields = m.fields
        #check for important data types
        mdata = {}
        for field in fields:
            if field.name in allowed_fields:
                if field.name=='timestamp':
                    mdata[field.name] = UTC.localize(field.value).astimezone(CST)
                else:
                    mdata[field.name] = field.value
        for rf in required_fields:
            if rf not in mdata:
                skip=True
        if not skip:
            data.append(mdata)
    #write to csv
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(allowed_fields)
        for entry in data:
            writer.writerow([ str(entry.get(k, '')) for k in allowed_fields])
    print('wrote %s' % output_file)

if __name__=='__main__':
    main()

