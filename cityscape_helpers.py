#!/usr/bin/env python
__copyright__ = """
Copyright (c) 2021 Tananaev Denis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this permission
notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from labels import labels


def get_color_palette(max_label=19, add_background=True):
    """
    The function uploads CityScape color palette.
    Arguments:
        max_label: numbers of labels
        add_background: if true adds black color for background class
    Returns:
        color_pallete: list with rgb colors
    """
    colors_dict = {label.trainId: [*label.color] for label in labels}
    color_palette = []
    for i in range(max_label):
        color_palette.append(colors_dict[i])
    if add_background:
        color_palette.append([0, 0, 0])
    return color_palette


def get_label_name(max_label=19, add_background=True):
    """
    The function uploads CityScape class names.
    Arguments:
        max_label: number of labels
        add_background: if true adds "background" class name
    Returns:
        name_list: list with names
    """
    name_dict = {label.trainId: label.name for label in labels}
    name_list = []
    for i in range(max_label):
        name_list.append(name_dict[i])
    if add_background:
        name_list.append("background")
    return name_list
