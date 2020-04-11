copy x64\Release\testbench.exe eternal_fortress\demo.exe
robocopy /E assets eternal_fortress\assets
robocopy /E shaders eternal_fortress\shaders
del eternal_fortress\assets\conference.obj
del eternal_fortress\assets\conference.mtl
del eternal_fortress\assets\aalto.png
del eternal_fortress\assets\MonValley_A_LookoutPoint_8k.jpg
