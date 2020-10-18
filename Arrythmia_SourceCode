clc;
load('100m.mat')
ECGsignal = val/200;
T_ECGsignal= transpose(ECGsignal);
Fs = 800;
t = (0:length(T_ECGsignal)-1)/Fs;
plot(t,T_ECGsignal)
coeffs = mfcc(T_ECGsignal,Fs,'NumCoeffs',3);
coeffs1=transpose(coeffs);
figure;
plot(coeffs1)
figure;
plot(t,coeffs1)
