"""
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Ontology Engineering Group
        http://www.oeg-upm.net/
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Copyright (C) 2017 Ontology Engineering Group.
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

from pyparsing import Word, alphas, alphanums, ZeroOrMore, Literal, ParseException

from agora_wot.blocks.operators import lslug, objectValue

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.wot')

lparen = Literal("(")
rparen = Literal(")")

identifier = Word(alphas)
functor = identifier

symbols = """!"#%&'*+,-./:;<=>?@[\]^_`|~"""
arg_chars = alphanums + symbols

# allow expression to be used recursively
arg = Word(arg_chars)
args = arg + ZeroOrMore("," + arg)

expression = functor + lparen + args + rparen

dollar = Literal("$")
param = Word(alphas)
rest = Word(arg_chars)
wrap_param = dollar + param + ZeroOrMore(rest)

operators = {
    'lslug': lslug,
    'object': objectValue
}


def evaluate_expression(expr, **kwargs):
    tokens = expression.parseString(expr)
    if tokens:
        f = tokens[0]
        args = tokens[2:-1]
        try:
            return unicode(operators[f](*args, **kwargs))
        except Exception, e:
            raise AttributeError(expr)


def find_params(expr):
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
    ev_string = string
    for expr in re.findall(r"\{\{([^}]+)\}\}", string):
        r_string = evaluate_expression(expr, **kwargs)
        if r_string is not None:
            ev_string = ev_string.replace('{{%s}}' % expr, r_string)

    return ev_string
