class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'fundus':
            #return '../../../../data/disc_cup_split/'  # foler that contains leftImg8bit/
            return'/data/AF-MFD/fundus'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
