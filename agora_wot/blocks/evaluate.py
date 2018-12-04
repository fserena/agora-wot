"""
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Copyright (C) 2018 Fernando Serena
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
"""
import logging
import re

from pyparsing import Word, alphas, alphanums, ZeroOrMore, Literal, ParseException, Forward

from agora_wot.blocks.operators import lslug, objectValue, filterObjects

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.wot')

lparen = Literal("(")
rparen = Literal(")")

identifier = Word(alphas)
functor = identifier

symbols = """!"#%&'*+,-./:;<=>?@[\]^_`|~"""
arg_chars = alphanums + symbols

# allow expression to be used recursively
expression = Forward()
arg = expression | Word(arg_chars)
args = arg + ZeroOrMore("," + arg)

expression << functor + lparen + args + rparen

dollar = Literal("$")
param = Word(alphas)
rest = Word(arg_chars)
wrap_param = dollar + param + ZeroOrMore(rest)


def evaluate_expression(expr, **kwargs):
    # type: (basestring, dict) -> unicode

    def context_w(f):
        def wrapper(*args, **ext_kwargs):
            return f(*args, **kwargs)

        return wrapper

    operators = {
        'lslug': context_w(lslug),
        'object': context_w(objectValue),
        'filter': context_w(filterObjects)
    }

    try:

        ret = eval(expr, {'__builtins__': None}, operators)
        return unicode(ret)
    except (IndexError, SyntaxError) as e:
        log.warning(e.message)


def find_params(expr):
    # type: (str) -> iter[basestring]
    try:
        first = True
        for part in expr.split('$'):
            if part:
                if not first or expr[0] == '$':
                    tokens = wrap_param.parseString('$' + part)
                    if tokens:
                        yield tokens[0] + tokens[1]
            first = False
    except ParseException, e:
        log.warn(e.message)


def evaluate(string, **kwargs):
    # type: (basestring, dict) -> basestring
    ev_string = string
    for expr in re.findall(r"\{\{([^}]+)\}\}", string):
        r_string = evaluate_expression(expr, **kwargs)
        if r_string is not None:
            ev_string = ev_string.replace('{{%s}}' % expr, r_string)

    return ev_string
