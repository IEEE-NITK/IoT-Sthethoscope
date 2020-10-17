clc; 
clear all;
close all;

Fs = 360;
td = 10;

signal = struct2array(load('100m'));
signal1 = signal(2:2, 1: Fs*td)/200;

t = [0: 1/Fs: td-(1/Fs)];
subplot(3,1,1);
plot(t, signal1);
title('Arrythmia. sample no:100');

z = abs(fftshift(fft(signal1))) / Fs;
f = [-Fs/2: 1/td: (Fs/2)-(1/td)];
subplot(3,1,2);
plot(f, z);

subplot(3,1,3);
spectrogram(signal1, 64, 16, 64, Fs, 'yaxis');

signal = struct2array(load('114m'));
signal2 = signal(1:1, 1: Fs*td)/200;

t = [0: 1/Fs: td-(1/Fs)];
subplot(3,1,1);
plot(t, signal2);
title('Arrythmia. sample no:114');

z = abs(fftshift(fft(signal2))) / Fs;
f = [-Fs/2: 1/td: (Fs/2)-(1/td)];
subplot(3,1,2);
plot(f, z);

subplot(3,1,3);
spectrogram(signal2, 64, 16, 64, Fs, 'yaxis');

[audio1, Fs] = audioread("breathing-deep-healthy.wav");
td = size(audio1)/Fs;
td = td(1:1);

t = [0: 1/Fs: td-(1/Fs)];
subplot(3,1,1);
plot(t, audio1);
title('covid status: healthy');

z = abs(fftshift(fft(audio1))) / Fs;
f = [-Fs/2: 1/td: (Fs/2)-(1/td)];
subplot(3,1,2);
plot(f, z);

subplot(3,1,3);
spectrogram(audio1, 2048, 512, 2048, Fs, 'yaxis');

[audio2, Fs] = audioread("breathing-deep-resp-illness-not-iden.wav");
td = size(audio2)/Fs;
td = td(1:1);

t = [0: 1/Fs: td-(1/Fs)];
subplot(3,1,1);
plot(t, audio2);
title('covid status: respiratory illness not identified');

z = abs(fftshift(fft(audio2))) / Fs;
f = [-Fs/2: 1/td: (Fs/2)-(1/td)];
subplot(3,1,2);
plot(f, z);

subplot(3,1,3);
spectrogram(audio2, 2048, 512, 2048, Fs, 'yaxis');




