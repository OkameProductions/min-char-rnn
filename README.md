# min-char-rnn
Andrey Karpathy (@karpathy)'s min-char-rnn.py, edited a bit to support reading different inputs and writing to files and to work with Python3.
## Usage
```python min-char-rnn.py input.txt output.txt```
## Parts of the code edited
New section added:
```py
# Code changes to support custom input-output by Scy Productions (@RealScyP)
if len(sys.argv) == 3:
  input = sys.argv[1]
  outputFile = sys.argv[2]
  output = open(outputFile, 'w')
elif len(sys.argv) == 2:
  input = sys.argv[1]
  outputFile = "output.txt"
  output = open(outputFile, 'w')
else:
  input = "input.txt"
  outputFile = "output.txt"
  output = open(outputFile, 'w')
```
Modified sections (not including modifications for Python3) (">" signifies modified or added line, the sign doesn't appear in the code):
```py
# data I/O
> data = open(input, 'r').read() # should be simple plain text file # ''input.txt'' was replaced with 'input' to support custom input-output
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
> print('data has %d characters, %d unique.' % (data_size, vocab_size), file=output)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
```
```py
  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    > print('----\n %s \n----' % (txt, ), file=output)
    print('----\n %s \n----' % (txt, ))
```
```py
  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0:
   > print('iter %d, loss: %f' % (n, smooth_loss), file=output) # print progress to file
   > print('iter %d, loss: %f' % (n, smooth_loss)) # print progress to console
 ```
