"""Class definition for the WikiExtractorByTemplate"""
import requests
import sys
import pandas as pd
from tqdm import tqdm

class WikiExtractorByTemplate:
    """Iterable object that fetches all the information that
    transcludes a given template from the wikipedia api."""

    def __init__(self, template_name: str, get_pages = False, pvipdays:int = 10, max_results: int = -1):
        template_name = template_name.replace(
                " ",
                "%20"
        )  # enforce proper url formatting
        max_results = "max" if max_results == -1 else max_results 


        self._template_name = template_name
        self._geicontinue: str   | None = None
        self._andcontinue: tuple | None = None
        self._done = False
        self._root = "https://en.wikipedia.org/w/api.php?"\
                     "action=query&"\
                     "format=json&"\
                     "prop=info" + f"{'%7Cpageviews' if pvipdays != -1 else ''}" + f"{'%7Crevisions&' if get_pages else '&'}"\
                     "generator=embeddedin&"\
                     "formatversion=2&"\
                     "inprop=url&"\
                     f"{'pvipdays=' + pvipdays + '&' if pvipdays != -1 else ''}"\
                     f"{'rvprop=content&rvslots=*&' if get_pages else ''}"\
                     f"geititle=Template%3A{template_name}&"\
                     f"{'' if get_pages else 'geilimit=' + max_results}"

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        url = self._root
        
        # update pagination
        if self._geicontinue is not None:
            url += f"&geicontinue={self._geicontinue}"
        if self._andcontinue is not None:
            url += f"&{self._andcontinue[0]}={self._andcontinue[1]}"

        # get request
        json_info = requests.get(url).json()

        # check for pagination
        if not json_info.get("continue", False):
            self._done = True
        else:
            if next_page := json_info["continue"].get("geicontinue", False):
                self._geicontinue = next_page
            else:
                continue_type = list(json_info["continue"].keys())[0]
                self._andcontinue = (continue_type, json_info["continue"][continue_type])

        return json_info


if __name__ == "__main__":
    # we can select article pages about humans by checking if the pages
    # transclude the 'Death date and age' template.
    if len(sys.argv) != 3:
        print("usage: [template_name] [output path]")
        quit()
    template, out_path = sys.argv[1:]

    ext = WikiExtractorByTemplate(template)

    #df = [item for batch in ext for item in batch]
    df = {}
    for batch in tqdm(ext):
        for item in batch["query"]["pages"]:
            if item['title'] in df:
                print("MEGA ERROR!!!!")
                print(item)
                print(len(df))
                breakpoint()
                quit()
            df[item['title']] = item
