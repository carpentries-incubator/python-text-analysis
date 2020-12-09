---
title: "Data Types and Type Conversion"
teaching: 10
exercises: 10
questions:
- "What kinds of data do programs store?"
- "How can I convert one type to another?"
objectives:
- "Explain key differences between integers and floating point numbers."
- "Explain key differences between numbers and character strings."
- "Use built-in functions to convert between integers, floating point numbers, and strings."
keypoints:
- "Every value has a type."
- "Use the built-in function `type` to find the type of a value."
- "Types control what operations can be done on values."
- "Strings can be added and multiplied."
- "Strings have a length (but numbers don't)."
- "Must convert numbers to strings or vice versa when operating on them."
- "Can mix integers and floats freely in operations."
- "Variables only change value when something is assigned to them."
---
## Every value has a type.

*   Every value in a program has a specific type.
*   Integer (`int`): represents positive or negative whole numbers like 3 or -512.
*   Floating point number (`float`): represents real numbers like 3.14159 or -2.5.
*   Character string (usually called "string", `str`): text.
    *   Written in either single quotes or double quotes (as long as they match).
    *   The quote marks aren't printed when the string is displayed.
    *   We will focus on strings since they are very important in text analysis.

## Use the built-in function `type` to find the type of a value.

*   Use the built-in function `type` to find out what type a value has.
*   Works on variables as well.
    *   But remember: the *value* has the type --- the *variable* is just a label.

~~~
print(type(52))
~~~
{: .language-python}
~~~
<class 'int'>
~~~
{: .output}

~~~
fitness = 'average'
print(type(fitness))
~~~
{: .language-python}
~~~
<class 'str'>
~~~
{: .output}

## Types control what operations (or methods) can be performed on a given value.

*   A value's type determines what the program can do to it.

~~~
print(5 - 3)
~~~
{: .language-python}
~~~
2
~~~
{: .output}

~~~
print('hello' - 'h')
~~~
{: .language-python}
~~~
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-2-67f5626a1e07> in <module>()
----> 1 print('hello' - 'h')

TypeError: unsupported operand type(s) for -: 'str' and 'str'
~~~
{: .error}

## You can use the "+" and "*" operators on strings.

*   "Adding" character strings concatenates them.

~~~
full_name = 'Ahmed' + ' ' + 'Walsh'
print(full_name)
~~~
{: .language-python}
~~~
Ahmed Walsh
~~~
{: .output}

*   Multiplying a character string by an integer _N_ creates a new string that consists of that character string repeated  _N_ times.
    *   Since multiplication is repeated addition.

~~~
separator = '=' * 10
print(separator)
~~~
{: .language-python}
~~~
==========
~~~
{: .output}

## Strings have a length (but numbers don't).

*   The built-in function `len` counts the number of characters in a string.

~~~
print(len(full_name))
~~~
{: .language-python}
~~~
11
~~~
{: .output}

*   But numbers don't have a length (not even zero).

~~~
print(len(52))
~~~
{: .language-python}
~~~
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-3-f769e8e8097d> in <module>()
----> 1 print(len(52))

TypeError: object of type 'int' has no len()
~~~
{: .error}

## <a name='convert-numbers-and-strings'></a> Must convert numbers to strings or vice versa when operating on them.

*   Cannot add numbers and strings.

~~~
print(1 + '2')
~~~
{: .language-python}
~~~
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-4-fe4f54a023c6> in <module>()
----> 1 print(1 + '2')

TypeError: unsupported operand type(s) for +: 'int' and 'str'
~~~
{: .error}

*   Not allowed because it's ambiguous: should `1 + '2'` be `3` or `'12'`?
*   Some types can be converted to other types by using the type name as a function.

~~~
print(1 + int('2'))
print(str(1) + '2')
~~~
{: .language-python}
~~~
3
12
~~~
{: .output}


## Variables only change value when something is assigned to them.

*   If we make one cell in a spreadsheet depend on another,
    and update the latter,
    the former updates automatically.
*   This does **not** happen in programming languages.

~~~
first = "Ahmed"
second = first + " J"
first = "Piper"
print('first is', first, 'and second is', second)
~~~
{: .language-python}
~~~
first is Piper and second is Ahmed J
~~~
{: .output}

*   The computer reads the value of `first` when doing the concatenation,
    creates a new value, and assigns it to `second`.
*   After that, `second` does not remember where it came from.

> ## What type is it?
>
> 1. What type of the value 3.4?
> 2. What is the type of "Alice"?
>
> How can you find out?
>
> > ## Solution
> >
> > 1. 3.4 it is a decimal number which is called a floating-point number (often abbreviated "float") in python.
> > 2. "Alice" is a string of characters (called "string") which is a common type of text.
> >
> > ~~~
> > print(type(3.4))
> > ~~~
> > {: .language-python}
> > ~~~
> > <class 'float'>
> > ~~~
> > {: .output}
> >
> >
> > ~~~
> > print(type("Alice"))
> > ~~~
> > {: .language-python}
> > ~~~
> > <class 'str'>
> > ~~~
> > {: .output}
> {: .solution}
{: .challenge}

> ##  Type Conversion
>
> 1. How would you convert a string `"42"` to be an integer?
> 2. What if it were an integer `42` and you wanted to make it a string?
>
> > ## Solution
> >
> > 1. You can convert a string into an integer using the `int()` function.
> > 2. To conver the integer into a string use the `str()` function.
> >
> > ~~~
> > int("42")
> > ~~~
> > {: .language-python}
> > ~~~
> > str(42)
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

> ## Choose a Type
>
> What type of value (integer, floating point number, or character string)
> would you use to represent each of the following?  Try to come up with more than one good answer for each problem.  For example, in  # 1, when would counting days with a floating point variable make more sense than using an integer?
>
> 1. Number of days since the start of the year.
> 4. A name
> 3. A book
> 5. Current population of a city.
> 6. Number of times a word occurred in a book.
> 7. Portion of an article which is stop words (“the”, “a”, “an”, “in”).
>
> > ## Solution
> >
> > The answers to the questions are:
> > 1. Integer, since the number of days would lie between 1 and 365.
> > 2. Character string as it contains letters and words.
> > 3. Character string as it contains letters and words.
> > 5. Choose floating point to represent population as large aggregates (eg millions), or integer to represent population in units of individuals.
> > 4. Integer, since this is likely to be a whole number.
> > 6. Floating point number, since an fraction of the whole.
> > {: .output}
> {: .solution}
{: .challenge}


> ## Practice with String Lengths
>
> How would you determine number of letters in the following string?
> ~~~
> my_quote = "Python is a fun language to learn!"
> ~~~
> {: .language-python}
>
> > ## Solution
> >
> > You can use the `len()` function on a string to determine the length.
> > You may need to add a `print()` statement to see the result depending on your python environment.
> >
> > ~~~
> > len(my_quote)
> > ~~~
> > {: .language-python}
> >
> > ~~~
> > 34
> > ~~~
> > {: .output}
> {: .solution}
{: .challenge}
