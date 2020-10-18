clc;
load('100m.mat')
ECGsignal = val/200;
T_ECGsignal= transpose(ECGsignal);
Fs = 3600;
t = (0:length(T_ECGsignal)-1)/Fs;
plot(t,T_ECGsignal)
coeffs = mfcc(T_ECGsignal,Fs,'NumCoeffs',13);
coeffs1=transpose(coeffs);
figure;
plot(-coeffs1)
figure;
imagesc( [1:size(coeffs1,2)], [0:13-1], coeffs1); 
           axis( 'xy' );
           xlabel( 'Frame index' ); 
           ylabel( 'Cepstrum index' );
           title( 'Mel frequency cepstrum' );
