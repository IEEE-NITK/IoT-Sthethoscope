clc;
clear all;
close all;
[ECGsignal, Fs] = audioread('breathing-deep.wav');
%ECGsignal= audioread('breathing-deep.wav');
%Fs = 3600;
t = (0:length(ECGsignal)-1)/Fs;
plot(t,ECGsignal)
coeffs = mfcc(ECGsignal,Fs,'NumCoeffs',13);
coeffs1=transpose(coeffs);
figure;
plot(-coeffs1)
figure;
imagesc( [1:size(coeffs1,2)], [0:13-1], coeffs1); 
           axis( 'xy' );
           xlabel( 'Frame index' ); 
           ylabel( 'Cepstrum index' );
           title( 'Mel frequency cepstrum' );