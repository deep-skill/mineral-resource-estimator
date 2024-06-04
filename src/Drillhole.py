# Código para hacer "desurveying" y obtener los datos de entrada del modelo (X, Y, Z, DIP, AZM)
# Desurveying: https://opengeostat.github.io/pygslib/Tutorial.html#desurveying

import numpy as np
from math import *

def ang2cart(azm, dip):
    DEG2RAD = 3.141592654 / 180.0

    razm = azm * DEG2RAD
    rdip = -dip * DEG2RAD

    x = sin(razm) * cos(rdip)
    y = cos(razm) * cos(rdip)
    z = sin(rdip)

    return x, y, z


def cart2ang(x, y, z):
    EPSLON = 1.0e-4

    if abs(x) + EPSLON > 1.001 or abs(y) + EPSLON > 1.001 or abs(z) + EPSLON > 1.001:
        print('Coordinate x, y or z is outside the interval [-1, 1]')

    if x > 1. : x = 1.
    if x < -1.: x = - 1.

    if y > 1. : y = 1.
    if y < -1.: y = - 1.

    if z > 1.: z = 1.
    if z < -1.: z = -1.

    pi = 3.141592654
    RAD2DEG = 180.0 / pi

    azm = atan2(x, y)

    if azm < 0:
        azm = azm + pi / 2

    azm = azm * RAD2DEG

    dip = -asin(z) * RAD2DEG

    return azm, dip

def interp_ang1D(azm1, dip1, azm2, dip2, len12, d1):
    x1, y1, z1 = ang2cart(azm1, dip1)
    x2, y2, z2 = ang2cart(azm2, dip2)

    x = x2*d1/len12 + x1*(len12-d1)/len12
    y = y2*d1/len12 + y1*(len12-d1)/len12
    z = z2*d1/len12 + z1*(len12-d1)/len12

    return cart2ang(x, y, z)

def dsmincurb(len12, azm1, dip1, azm2, dip2):
    DEG2RAD = 3.141592654/180.0

    i1 = (90 - dip1) * DEG2RAD
    a1 = azm1 * DEG2RAD

    i2 = (90 - dip2) * DEG2RAD
    a2 = azm2 * DEG2RAD

    dl = acos(cos(i2-i1)-sin(i1)*sin(i2)*(1-cos(a2-a1)))

    if dl != 0.:
        rf = 2 * tan(dl / 2) / dl
    else:
        rf = 1

    dz = 0.5*len12*(cos(i1)+cos(i2))*rf
    dn = 0.5*len12*(sin(i1)*cos(a1)+sin(i2)*cos(a2))*rf
    de = 0.5*len12*(sin(i1)*sin(a1)+sin(i2)*sin(a2))*rf

    return dz, dn, de

def dstangential(len12, azm1, dip1):
    DEG2RAD = 3.141592654/180.0

    i1 = (90 - dip1) * DEG2RAD
    a1 = azm1 * DEG2RAD

    dz = len12*cos(i1)
    dn = len12*sin(i1)*cos(a1)
    de = len12*sin(i1)*sin(a1)

    return dz,dn,de


class Drillhole:
    def __init__(self, collar, survey):
        self.collar = collar
        self.survey = survey
        self.table = {}

    def add_table(self, table, table_name):
        if table_name not in self.table:
            self.table[table_name] = table.copy(deep = True)

        self.table[table_name]['FROM'] = self.table[table_name]['FROM'].astype(float)
        self.table[table_name]['TO'] = self.table[table_name]['TO'].astype(float)
        self.table[table_name]['BHID'] = self.table[table_name]['BHID'].astype(str)

        self.table[table_name]['BHID']= self.table[table_name]['BHID'].str.upper()

        self.table[table_name].sort_values(by=['BHID', 'FROM'], inplace=True)

        self.table[table_name].reset_index(level=None, drop=True, inplace=True, col_level=0, col_fill='')

    def desurvey_survey(self, method=1):
        self.survey['x'] = self.survey['y'] = self.survey['z'] = np.nan

        self.survey.sort_values(by = ['BHID', 'AT'])

        for c in self.collar['BHID']:
            XC,YC,ZC = self.collar.loc[self.collar['BHID']==c, ['XCOLLAR','YCOLLAR','ZCOLLAR']].values[0]

            AT = self.survey.loc[self.survey['BHID']==c, 'AT'].values
            DIP = self.survey.loc[self.survey['BHID']==c, 'DIP'].values
            AZ = self.survey.loc[self.survey['BHID']==c, 'AZ'].values

            dz = np.empty(AT.shape)
            dn = np.empty(AT.shape)
            de = np.empty(AT.shape)

            x = np.empty(AT.shape)
            y = np.empty(AT.shape)
            z = np.empty(AT.shape)

            dz[0] = dn[0] = de[0] = 0
            x[0] = XC
            y[0] = YC
            z[0] = ZC

            for i in range(1, AT.shape[0]):
                if method == 1:
                    dz[i],dn[i],de[i] = dsmincurb(len12 = AT[i] - AT[i-1], azm1 = AZ[i-1],  dip1 = DIP[i-1], azm2=AZ[i], dip2 = DIP[i])
                else:
                    dz[i],dn[i],de[i] = dstangential(len12 = AT[i] - AT[i-1], azm1 = AZ[i-1],  dip1 = DIP[i-1])

                x[i] = x[i-1] + de[i]
                y[i] = y[i-1] + dn[i]
                z[i] = z[i-1] - dz[i]

            self.survey.loc[self.survey['BHID']==c,'x'] = x
            self.survey.loc[self.survey['BHID']==c,'y'] = y
            self.survey.loc[self.survey['BHID']==c,'z'] = z

    def desurvey_table(self, table_name, method=1):
        if 'x' not in self.survey.columns or \
           'y' not in self.survey.columns or \
           'z' not in self.survey.columns:

           self.desurvey_survey(method = method)

        table = self.table[table_name].set_index('BHID')
        table ['xb'] = np.nan
        table ['yb'] = np.nan
        table ['zb'] = np.nan
        table ['xe'] = np.nan
        table ['ye'] = np.nan
        table ['ze'] = np.nan
        table ['xm'] = np.nan
        table ['ym'] = np.nan
        table ['zm'] = np.nan
        table ['azmb'] = np.nan
        table ['dipb'] = np.nan
        table ['azme'] = np.nan
        table ['dipe'] = np.nan
        table ['azmm'] = np.nan
        table ['dipm'] = np.nan

        survey = self.survey.set_index('BHID')

        for c in self.table[table_name]['BHID'].unique():

            # get survey
            AT =  survey.loc[c, ['AT']].values.ravel()
            DIP = survey.loc[c, ['DIP']].values.ravel()
            AZ =  survey.loc[c, ['AZ']].values.ravel()
            xs =  survey.loc[c, ['x']].values.ravel()
            ys =  survey.loc[c, ['y']].values.ravel()
            zs =  survey.loc[c, ['z']].values.ravel()

            # get from, to, y mid interval
            db = table.loc[c, ['FROM']].values.ravel() # [[]].values.ravel() is to prevent getting a scalar if shape is 1
            de = table.loc[c, ['TO']].values.ravel()
            dm = db + (de-db)/2

            # add at the end of the survey if de< AT
            if de[-1]>= AT[-1]:
                AZ = np.append(AZ, AZ[-1])
                DIP = np.append(DIP, DIP[-1])
                AT = np.append(AT, de[-1] + 0.01)

            #get the index where each interval is located
            jb = np.searchsorted(AT, db, side='right')
            je = np.searchsorted(AT, de, side='right')
            jm = np.searchsorted(AT, dm, side='right')


            azmt = np.empty(jb.shape)
            dipt = np.empty(jb.shape)
            x = np.empty(jb.shape)
            y = np.empty(jb.shape)
            z = np.empty(jb.shape)

            # the bigining
            for i in range(jb.shape[0]):
                d1 = db[i] -AT[jb[i]-1]
                lll1 = AT[jb[i]]
                lll2 = AT[jb[i]-1]
                len12 = lll1-lll2
                azm1 = AZ[jb[i]-1]
                dip1 = DIP[jb[i]-1]
                azm2 = AZ[jb[i]]
                dip2 = DIP[jb[i]]
                azmt[i],dipt[i] = interp_ang1D(azm1, dip1, azm2, dip2, len12, d1)
                if method==1:
                    dz,dy,dx = dsmincurb(d1, azm1,  dip1, azmt[i], dipt[i])
                else:
                    dz,dy,dx = dstangential(d1, azm1,  dip1)

                x[i] = dx + xs[jb[i]-1]
                y[i] = dy + ys[jb[i]-1]
                z[i] = zs[jb[i]-1] - dz

            table.loc[c,'azmb']  = azmt
            table.loc[c,'dipb']  = dipt
            table.loc[c,'xb']  = x
            table.loc[c,'yb']  = y
            table.loc[c,'zb']  = z

            for i in range(je.shape[0]):
                d1 = de[i] -AT[je[i]-1]
                len12 = AT[je[i]]-AT[je[i]-1]
                azm1 = AZ[je[i]-1]
                dip1 = DIP[je[i]-1]
                azm2 = AZ[je[i]]
                dip2 = DIP[je[i]]
                azmt[i],dipt[i] = interp_ang1D(azm1, dip1, azm2, dip2, len12, d1)
                if method==1:
                    dz,dy,dx = dsmincurb(d1, azm1,  dip1, azmt[i], dipt[i])
                else:
                    dz,dy,dx = dstangential(d1, azm1,  dip1)
                x[i] = dx + xs[je[i]-1]
                y[i] = dy + ys[je[i]-1]
                z[i] = zs[je[i]-1] - dz

            table.loc[c,'azme']  = azmt
            table.loc[c,'dipe']  = dipt
            table.loc[c,'xe']  = x
            table.loc[c,'ye']  = y
            table.loc[c,'ze']  = z

            for i in range(jm.shape[0]):
                d1 = dm[i] -AT[jm[i]-1]
                len12 = AT[jm[i]]-AT[jm[i]-1]
                azm1 = AZ[jm[i]-1]
                dip1 = DIP[jm[i]-1]
                azm2 = AZ[jm[i]]
                dip2 = DIP[jm[i]]
                azmt[i],dipt[i] = interp_ang1D(azm1, dip1, azm2, dip2, len12, d1)
                if method==1:
                    dz,dy,dx = dsmincurb(d1, azm1,  dip1, azmt[i], dipt[i])
                else:
                    dz,dy,dx = dstangential(d1, azm1,  dip1)

                x[i] = dx + xs[jm[i]-1]
                y[i] = dy + ys[jm[i]-1]
                z[i] = zs[jm[i]-1] - dz


            table.loc[c,'azmm']  = azmt
            table.loc[c,'dipm']  = dipt
            table.loc[c,'xm']  = x
            table.loc[c,'ym']  = y
            table.loc[c,'zm']  = z

        self.table[table_name] = table.reset_index()

